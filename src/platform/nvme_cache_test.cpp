// src/platform/nvme_cache_test.cpp - Unit tests for NvmeCache storage engine.
#include "strata/platform/nvme_cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

static int fails = 0;
#define CHECK(c) do { \
    if (!(c)) { \
        std::printf("FAIL line %d: %s\n", __LINE__, #c); \
        ++fails; \
    } \
} while (0)

namespace {

using namespace strata::core;
using namespace strata::platform;

struct TempDir {
    std::filesystem::path path;
    TempDir(const std::string& prefix = "strata_nvme_test_") {
        auto base = std::filesystem::temp_directory_path();
        static std::atomic<uint64_t> counter = 1;
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / (prefix + std::to_string(now) + "_" + std::to_string(counter++));
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

KvBlock make_test_block(uint64_t seed, size_t num_tokens = 128, int format = kKvInt8, uint64_t parent_hash = 0) {
    KvBlock block;
    block.header.format = format;
    block.header.block_tokens = kKvBlockTokens;
    block.header.num_layers = 1;
    block.header.num_heads = 2;
    block.header.head_dim = 256;
    block.header.parent_hash = parent_hash;

    block.payload.token_ids.resize(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        block.payload.token_ids[i] = static_cast<int32_t>(seed * 1000 + i);
    }

    const size_t byte_count = num_tokens * 64;
    block.payload.k_data.resize(byte_count);
    block.payload.v_data.resize(byte_count);
    for (size_t i = 0; i < byte_count; ++i) {
        block.payload.k_data[i] = static_cast<uint8_t>((seed + i) & 0xFF);
        block.payload.v_data[i] = static_cast<uint8_t>((seed + i * 3) & 0xFF);
    }

    block.header.token_count = static_cast<uint32_t>(num_tokens);
    block.header.block_hash = kv_block_hash(block);

    // Canonicalize header fields via serialize roundtrip
    AlignedBuffer buf;
    kv_block_serialize(block, buf);
    KvBlock canonical;
    kv_block_deserialize(buf, canonical, false);
    return canonical;
}

// -----------------------------------------------------------------------------
// Test 1: Async write followed by async read, verify data matches 100%
// -----------------------------------------------------------------------------
void test_async_write_read_roundtrip() {
    std::printf("[Test 1] Async write followed by async read roundtrip...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 2;
    config.direct_io = true;
    config.auto_direct_io_fallback = true;

    NvmeCache cache(config);

    const auto block = make_test_block(101, 64, kKvInt8, 0x11223344);
    const uint64_t hash = block.header.block_hash;

    std::atomic<bool> write_done = false;
    std::atomic<bool> write_ok = false;
    std::string write_err;

    cache.write_block_async(block, [&](bool ok, const std::string& err) {
        write_ok = ok;
        write_err = err;
        write_done = true;
    });

    cache.flush_sync();
    CHECK(write_done);
    CHECK(write_ok);
    CHECK(write_err.empty());
    CHECK(cache.contains(hash));

    // Verify async read
    std::atomic<bool> read_done = false;
    std::atomic<bool> read_ok = false;
    std::string read_err;
    KvBlock restored_block;

    cache.read_block_async(hash, [&](bool ok, KvBlock blk, const std::string& err) {
        read_ok = ok;
        read_err = err;
        restored_block = std::move(blk);
        read_done = true;
    });

    cache.flush_sync();
    CHECK(read_done);
    CHECK(read_ok);
    CHECK(read_err.empty());

    // 100% data preservation check
    CHECK(restored_block == block);

    // Also test synchronous wrappers on a separate block
    const auto block_sync = make_test_block(202, 128, kKvHybrid, hash);
    const uint64_t hash_sync = block_sync.header.block_hash;
    std::string sync_err;
    bool w_ok = cache.write_block_sync(block_sync, true, &sync_err);
    CHECK(w_ok);
    CHECK(sync_err.empty());
    CHECK(cache.contains(hash_sync));
    CHECK(cache.is_pinned(hash_sync));

    KvBlock restored_sync;
    bool r_ok = cache.read_block_sync(hash_sync, restored_sync, &sync_err);
    CHECK(r_ok);
    CHECK(sync_err.empty());
    CHECK(restored_sync == block_sync);
}

// -----------------------------------------------------------------------------
// Test 2: Concurrent async write/read queue stress test with callback completions
// -----------------------------------------------------------------------------
void test_concurrent_async_stress() {
    std::printf("[Test 2] Concurrent async write/read queue stress test...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 4;

    NvmeCache cache(config);

    constexpr size_t kNumBlocks = 32;
    std::vector<KvBlock> blocks;
    blocks.reserve(kNumBlocks);
    for (size_t i = 0; i < kNumBlocks; ++i) {
        blocks.push_back(make_test_block(1000 + i, 32 + (i % 96)));
    }

    std::atomic<size_t> write_completions = 0;
    std::atomic<size_t> write_failures = 0;

    // Queue writes across multiple concurrent threads
    std::vector<std::thread> client_threads;
    constexpr size_t kClientThreads = 4;
    for (size_t t = 0; t < kClientThreads; ++t) {
        client_threads.emplace_back([&, t]() {
            for (size_t i = t; i < kNumBlocks; i += kClientThreads) {
                cache.write_block_async(blocks[i], [&](bool ok, const std::string& /*err*/) {
                    if (ok) {
                        ++write_completions;
                    } else {
                        ++write_failures;
                    }
                });
            }
        });
    }

    for (auto& th : client_threads) {
        th.join();
    }
    client_threads.clear();

    cache.flush_sync();
    CHECK(write_completions == kNumBlocks);
    CHECK(write_failures == 0);
    CHECK(cache.get_stats().block_count == kNumBlocks);

    // Queue concurrent reads
    std::atomic<size_t> read_completions = 0;
    std::atomic<size_t> read_failures = 0;
    std::vector<KvBlock> restored_blocks(kNumBlocks);

    for (size_t t = 0; t < kClientThreads; ++t) {
        client_threads.emplace_back([&, t]() {
            for (size_t i = t; i < kNumBlocks; i += kClientThreads) {
                const uint64_t hash = blocks[i].header.block_hash;
                cache.read_block_async(hash, [&, i](bool ok, KvBlock blk, const std::string& /*err*/) {
                    if (ok) {
                        restored_blocks[i] = std::move(blk);
                        ++read_completions;
                    } else {
                        ++read_failures;
                    }
                });
            }
        });
    }

    for (auto& th : client_threads) {
        th.join();
    }

    cache.flush_sync();
    CHECK(read_completions == kNumBlocks);
    CHECK(read_failures == 0);

    for (size_t i = 0; i < kNumBlocks; ++i) {
        CHECK(restored_blocks[i] == blocks[i]);
    }
}

// -----------------------------------------------------------------------------
// Test 3: Manifest persistence across cache reopen
// -----------------------------------------------------------------------------
void test_manifest_persistence() {
    std::printf("[Test 3] Manifest persistence across cache reopen...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 2;

    constexpr size_t kBlockCount = 5;
    std::vector<KvBlock> written_blocks;
    written_blocks.reserve(kBlockCount);

    uint64_t expected_total_bytes = 0;

    {
        NvmeCache cache1(config);
        for (size_t i = 0; i < kBlockCount; ++i) {
            bool pin = (i % 2 == 0); // blocks 0, 2, 4 pinned
            size_t tokens = 20 + i * 20;
            auto blk = make_test_block(5000 + i, tokens, kKvInt8);
            written_blocks.push_back(blk);

            bool ok = cache1.write_block_sync(blk, pin);
            CHECK(ok);
        }
        cache1.flush_sync();
        expected_total_bytes = cache1.get_stats().total_bytes;
        CHECK(cache1.get_stats().block_count == kBlockCount);
        CHECK(cache1.get_stats().pinned_count == 3);
        // cache1 destroyed here; manifest is flushed to disk
    }

    // Verify manifest file exists on disk
    auto manifest_file = temp.path / ".strata_kv_manifest.bin";
    CHECK(std::filesystem::exists(manifest_file));
    CHECK(std::filesystem::file_size(manifest_file) > sizeof(ManifestHeader));

    // Reopen cache from the exact same directory
    {
        NvmeCache cache2(config);

        // Verify metadata is restored without re-reading payloads
        CHECK(cache2.get_stats().block_count == kBlockCount);
        CHECK(cache2.get_stats().pinned_count == 3);
        CHECK(cache2.get_stats().total_bytes == expected_total_bytes);

        for (size_t i = 0; i < kBlockCount; ++i) {
            const auto& orig = written_blocks[i];
            const uint64_t hash = orig.header.block_hash;

            CHECK(cache2.contains(hash));
            auto meta = cache2.get_metadata(hash);
            CHECK(meta.has_value());
            CHECK(meta->block_hash == hash);
            CHECK(meta->num_tokens == orig.header.token_count);
            CHECK(meta->is_pinned == (i % 2 == 0 ? 1u : 0u));
            CHECK(meta->file_size > 0);
        }

        // Verify payloads can now be read and match
        for (size_t i = 0; i < kBlockCount; ++i) {
            KvBlock restored;
            bool ok = cache2.read_block_sync(written_blocks[i].header.block_hash, restored);
            CHECK(ok);
            CHECK(restored == written_blocks[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Test 4: Disk budget and LRU eviction
// -----------------------------------------------------------------------------
void test_disk_budget_and_lru_eviction() {
    std::printf("[Test 4] Disk budget and LRU eviction policy...\n");
    TempDir temp;

    // Budget fits exactly 2 blocks (each block is padded to 4096 bytes)
    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.disk_budget_bytes = 2 * kKvBlockAlignment; // 8192 bytes
    config.num_worker_threads = 1;

    NvmeCache cache(config);

    auto b1 = make_test_block(7001, 10);
    auto b2 = make_test_block(7002, 10);
    auto b3 = make_test_block(7003, 10);
    auto b4 = make_test_block(7004, 10);

    // 1. Write Block 1 (pinned)
    bool ok1 = cache.write_block_sync(b1, true /* pinned */);
    CHECK(ok1);
    CHECK(cache.contains(b1.header.block_hash));
    CHECK(cache.is_pinned(b1.header.block_hash));

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // 2. Write Block 2 (unpinned) -> total is now 8192 (at budget)
    bool ok2 = cache.write_block_sync(b2, false /* unpinned */);
    CHECK(ok2);
    CHECK(cache.contains(b2.header.block_hash));
    CHECK(!cache.is_pinned(b2.header.block_hash));

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // 3. Write Block 3 (unpinned) -> exceeds budget!
    // Block 1 is pinned so it cannot be evicted.
    // Block 2 is unpinned and oldest -> must be evicted!
    bool ok3 = cache.write_block_sync(b3, false /* unpinned */);
    CHECK(ok3);

    CHECK(cache.contains(b1.header.block_hash)); // Pinned block 1 retained
    CHECK(!cache.contains(b2.header.block_hash)); // Unpinned block 2 evicted!
    CHECK(cache.contains(b3.header.block_hash)); // New block 3 present

    auto stats = cache.get_stats();
    CHECK(stats.evictions == 1);
    CHECK(stats.bytes_evicted == kKvBlockAlignment);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // 4. Write Block 4 (unpinned) -> exceeds budget again!
    // Block 1 is pinned -> retained!
    // Block 3 is unpinned and oldest -> evicted!
    bool ok4 = cache.write_block_sync(b4, false /* unpinned */);
    CHECK(ok4);

    CHECK(cache.contains(b1.header.block_hash)); // Pinned block 1 STILL intact!
    CHECK(!cache.contains(b3.header.block_hash)); // Block 3 evicted!
    CHECK(cache.contains(b4.header.block_hash)); // Block 4 present

    stats = cache.get_stats();
    CHECK(stats.evictions == 2);
    CHECK(stats.bytes_evicted == 2 * kKvBlockAlignment);

    // Verify pinned Block 1 is still completely readable and untampered
    KvBlock read_b1;
    bool read_ok = cache.read_block_sync(b1.header.block_hash, read_b1);
    CHECK(read_ok);
    CHECK(read_b1 == b1);
}

// -----------------------------------------------------------------------------
// Test 5: Error handling & corrupted block detection
// -----------------------------------------------------------------------------
void test_error_handling() {
    std::printf("[Test 5] Error handling and corrupt file detection...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 2;

    NvmeCache cache(config);

    // 1. Reading a non-existent block must fail with error
    const uint64_t bogus_hash = 0xDEADBEEFCAFE0001ULL;
    KvBlock out_blk;
    std::string err_msg;
    bool ok = cache.read_block_sync(bogus_hash, out_blk, &err_msg);
    CHECK(!ok);
    CHECK(!err_msg.empty());
    CHECK(cache.get_stats().read_misses > 0);

    // 2. Write a valid block
    auto valid_block = make_test_block(8888, 50);
    const uint64_t valid_hash = valid_block.header.block_hash;
    bool w_ok = cache.write_block_sync(valid_block);
    CHECK(w_ok);

    // Read back clean block
    bool r_ok = cache.read_block_sync(valid_hash, out_blk, &err_msg);
    CHECK(r_ok);
    CHECK(out_blk == valid_block);

    // 3. Corrupt file on disk
    auto block_path = cache.get_block_path(valid_hash);
    CHECK(std::filesystem::exists(block_path));

    FILE* fp = std::fopen(block_path.string().c_str(), "r+b");
    CHECK(fp != nullptr);
    if (fp) {
        // Corrupt magic or payload
        uint8_t garbage[16] = {0xFF, 0x00, 0xAA, 0x55, 0x12, 0x34};
        std::fseek(fp, 0, SEEK_SET);
        std::fwrite(garbage, 1, sizeof(garbage), fp);
        std::fclose(fp);
    }

    // Attempting to read corrupted block must be detected and rejected
    KvBlock bad_block;
    std::string corrupt_err;
    bool corrupt_ok = cache.read_block_sync(valid_hash, bad_block, &corrupt_err);
    CHECK(!corrupt_ok);
    CHECK(!corrupt_err.empty());
}

// -----------------------------------------------------------------------------
// Test 6: Clear operation and in-flight query coherence
// -----------------------------------------------------------------------------
void test_clear_and_in_flight() {
    std::printf("[Test 6] Clear operation and in-flight queries...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 2;

    NvmeCache cache(config);

    auto b1 = make_test_block(9001, 32);
    auto b2 = make_test_block(9002, 32);
    const uint64_t h1 = b1.header.block_hash;
    const uint64_t h2 = b2.header.block_hash;

    // Asynchronously queue write b1 without flushing immediately
    cache.write_block_async(b1);
    CHECK(cache.contains(h1)); // In-flight query must find it
    auto meta1 = cache.get_metadata(h1);
    CHECK(meta1.has_value());
    CHECK(meta1->block_hash == h1);

    cache.write_block_sync(b2);
    cache.flush_sync();

    CHECK(cache.contains(h1));
    CHECK(cache.contains(h2));
    CHECK(cache.get_stats().block_count == 2);
    CHECK(std::filesystem::exists(cache.get_block_path(h1)));
    CHECK(std::filesystem::exists(cache.get_block_path(h2)));

    // Clear cache (must not deadlock!)
    cache.clear();

    CHECK(cache.get_stats().block_count == 0);
    CHECK(cache.get_stats().total_bytes == 0);
    CHECK(!cache.contains(h1));
    CHECK(!cache.contains(h2));
    CHECK(!std::filesystem::exists(cache.get_block_path(h1)));
    CHECK(!std::filesystem::exists(cache.get_block_path(h2)));
}

// -----------------------------------------------------------------------------
// Test 7: CAS idempotency and pin status update on duplicate write
// -----------------------------------------------------------------------------
void test_cas_idempotency_and_pin_update() {
    std::printf("[Test 7] CAS idempotency and pin status update...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 2;

    NvmeCache cache(config);

    auto block = make_test_block(9100, 64);
    const uint64_t hash = block.header.block_hash;

    // 1. Initial write as unpinned
    bool ok1 = cache.write_block_sync(block, false);
    CHECK(ok1);
    CHECK(cache.contains(hash));
    CHECK(!cache.is_pinned(hash));
    CHECK(cache.get_stats().block_count == 1);
    CHECK(cache.get_stats().pinned_count == 0);

    // 2. Duplicate write with is_pinned = true
    bool ok2 = cache.write_block_sync(block, true);
    CHECK(ok2);
    CHECK(cache.is_pinned(hash)); // Pin status upgraded
    CHECK(cache.get_stats().block_count == 1); // Not duplicated
    CHECK(cache.get_stats().pinned_count == 1);

    // 3. Duplicate write again (already pinned, remains pinned)
    bool ok3 = cache.write_block_sync(block, false);
    CHECK(ok3);
    CHECK(cache.is_pinned(hash)); // Still pinned
    CHECK(cache.get_stats().block_count == 1);
}

// -----------------------------------------------------------------------------
// Test 8: Manifest crash recovery from .tmp and corrupt manifest detection
// -----------------------------------------------------------------------------
void test_manifest_crash_recovery() {
    std::printf("[Test 8] Manifest crash recovery and corruption rejection...\n");
    TempDir temp;

    NvmeCacheConfig config;
    config.cache_dir = temp.path;
    config.num_worker_threads = 2;

    auto b = make_test_block(9200, 48);
    const uint64_t hash = b.header.block_hash;

    {
        NvmeCache cache(config);
        CHECK(cache.write_block_sync(b));
        cache.flush_sync();
    }

    auto mpath = temp.path / ".strata_kv_manifest.bin";
    auto tmp_path = temp.path / ".strata_kv_manifest.bin.tmp";
    CHECK(std::filesystem::exists(mpath));

    // Simulate crash after writing .tmp but before rename
    std::error_code ec;
    std::filesystem::rename(mpath, tmp_path, ec);
    CHECK(!std::filesystem::exists(mpath));
    CHECK(std::filesystem::exists(tmp_path));

    // Reopen cache: must recover from .tmp
    {
        NvmeCache cache(config);
        CHECK(cache.contains(hash));
        CHECK(cache.get_stats().block_count == 1);
        KvBlock restored;
        CHECK(cache.read_block_sync(hash, restored));
        CHECK(restored == b);
    }
    CHECK(std::filesystem::exists(mpath));

    // Corrupt manifest file (invalid magic)
    {
        FILE* fp = std::fopen(mpath.string().c_str(), "wb");
        CHECK(fp != nullptr);
        if (fp) {
            uint8_t garbage[64] = {0};
            std::fwrite(garbage, 1, sizeof(garbage), fp);
            std::fclose(fp);
        }

        NvmeCache corrupt_cache(config);
        CHECK(corrupt_cache.get_stats().block_count == 0); // Safely rejected without crash
    }
}

} // namespace

int main() {
    std::printf("Running nvme_cache_test...\n");
    test_async_write_read_roundtrip();
    test_concurrent_async_stress();
    test_manifest_persistence();
    test_disk_budget_and_lru_eviction();
    test_error_handling();
    test_clear_and_in_flight();
    test_cas_idempotency_and_pin_update();
    test_manifest_crash_recovery();

    if (fails == 0) {
        std::printf("ALL NVME CACHE TESTS PASSED cleanly.\n");
        return 0;
    } else {
        std::printf("NVME CACHE TESTS FAILED: %d failures\n", fails);
        return 1;
    }
}
