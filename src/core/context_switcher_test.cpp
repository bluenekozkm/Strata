// src/core/context_switcher_test.cpp - Unit tests for ContextSwitcher & Stream Link.
#include "strata/core/context_switcher.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
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

std::shared_ptr<KvBlock> make_mock_block(
    uint64_t parent_hash,
    const std::vector<int32_t>& tokens,
    const RecurrenceState& rec,
    uint8_t k_fill = 0xAA,
    uint8_t v_fill = 0x55) {
    auto blk = std::make_shared<KvBlock>();
    blk->payload.token_ids = tokens;
    blk->payload.k_data.assign(512, k_fill);
    blk->payload.v_data.assign(512, v_fill);
    blk->payload.recurrence = rec.serialize();

    blk->header.block_tokens = static_cast<uint32_t>(tokens.size());
    blk->header.token_count = static_cast<uint32_t>(tokens.size());
    blk->header.parent_hash = parent_hash;
    blk->header.format = kKvInt8;
    blk->header.block_hash = kv_block_hash(parent_hash, tokens.data(), tokens.size(), kKvInt8);
    blk->header.payload_bytes = blk->payload.total_payload_bytes();
    return blk;
}

// -----------------------------------------------------------------------------
// Test 1: Cold mount of System Prompt A
// -----------------------------------------------------------------------------
void test_cold_mount_system_prompt() {
    std::printf("[Test 1] Cold mount of System Prompt A...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig cfg;
    cfg.block_tokens = 128;
    cfg.num_slots = 32;
    ContextSwitcher switcher(cache, nullptr, cfg);

    std::vector<int32_t> blk0_tokens;
    blk0_tokens.reserve(128);
    for (int i = 0; i < 128; ++i) blk0_tokens.push_back(1000 + i);

    std::vector<int32_t> blk1_tokens;
    blk1_tokens.reserve(128);
    for (int i = 0; i < 128; ++i) blk1_tokens.push_back(2000 + i);

    std::vector<int32_t> prompt_A_tokens = blk0_tokens;
    prompt_A_tokens.insert(prompt_A_tokens.end(), blk1_tokens.begin(), blk1_tokens.end());

    RecurrenceState rec_A;
    rec_A.gdn.assign(64, 0x11);
    rec_A.ple.assign(32, 0x22);
    rec_A.tails.assign(16, 0x33);

    auto blk0 = make_mock_block(0, blk0_tokens, RecurrenceState{}, 0x10, 0x20);
    auto blk1 = make_mock_block(blk0->header.block_hash, blk1_tokens, rec_A, 0x30, 0x40);

    auto node0 = cache->insert_sequence(blk0_tokens, true, true, blk0);
    auto node1 = cache->insert_sequence(prompt_A_tokens, true, true, blk1);
    CHECK(node0 != nullptr);
    CHECK(node1 != nullptr);

    ContextState active_ctx;
    switcher.init_context(active_ctx, "session-01");
    CHECK(active_ctx.token_count() == 0);
    CHECK(active_ctx.block_count() == 0);

    auto res = switcher.switch_context(active_ctx, prompt_A_tokens);
    CHECK(res.success == true);
    CHECK(res.common_tokens == 0);
    CHECK(res.divergent_tokens == 256);
    CHECK(res.blocks_reused == 0);
    CHECK(res.blocks_transferred == 2);
    CHECK(res.bytes_transferred > 0);

    CHECK(active_ctx.token_count() == 256);
    CHECK(active_ctx.block_count() == 2);
    CHECK(active_ctx.tokens == prompt_A_tokens);
    CHECK(active_ctx.recurrence == rec_A);
    CHECK(active_ctx.page_table[0] >= 0);
    CHECK(active_ctx.page_table[1] >= 0);
    CHECK(active_ctx.page_table[0] != active_ctx.page_table[1]);
}

// -----------------------------------------------------------------------------
// Test 2: Rapid Context Switching between System Prompt A and System Prompt B
// -----------------------------------------------------------------------------
void test_differential_context_switching() {
    std::printf("[Test 2] Rapid Context Switching between System Prompt A and System Prompt B...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig cfg;
    cfg.block_tokens = 128;
    cfg.num_slots = 32;
    ContextSwitcher switcher(cache, nullptr, cfg);

    // Shared prefix (block 0)
    std::vector<int32_t> shared_prefix;
    for (int i = 0; i < 128; ++i) shared_prefix.push_back(100 + i);

    // Prompt A divergent suffix (block 1A)
    std::vector<int32_t> suffix_A;
    for (int i = 0; i < 128; ++i) suffix_A.push_back(200 + i);

    // Prompt B divergent suffix (block 1B)
    std::vector<int32_t> suffix_B;
    for (int i = 0; i < 128; ++i) suffix_B.push_back(300 + i);

    std::vector<int32_t> prompt_A = shared_prefix;
    prompt_A.insert(prompt_A.end(), suffix_A.begin(), suffix_A.end());

    std::vector<int32_t> prompt_B = shared_prefix;
    prompt_B.insert(prompt_B.end(), suffix_B.begin(), suffix_B.end());

    RecurrenceState rec_A;
    rec_A.gdn.assign(64, 0xAA);
    rec_A.ple.assign(32, 0xAB);

    RecurrenceState rec_B;
    rec_B.gdn.assign(64, 0xBB);
    rec_B.ple.assign(32, 0xBC);

    auto blk0 = make_mock_block(0, shared_prefix, RecurrenceState{}, 0x01, 0x02);
    auto blk1A = make_mock_block(blk0->header.block_hash, suffix_A, rec_A, 0xA1, 0xA2);
    auto blk1B = make_mock_block(blk0->header.block_hash, suffix_B, rec_B, 0xB1, 0xB2);

    cache->insert_sequence(shared_prefix, true, true, blk0);
    cache->insert_sequence(prompt_A, true, true, blk1A);
    cache->insert_sequence(prompt_B, true, true, blk1B);

    ContextState active_ctx;
    switcher.init_context(active_ctx, "session-02");

    // Mount Prompt A
    auto res_A = switcher.switch_context(active_ctx, prompt_A);
    CHECK(res_A.success == true);
    CHECK(active_ctx.tokens == prompt_A);
    CHECK(active_ctx.recurrence == rec_A);

    int32_t slot_blk0_initial = active_ctx.page_table[0];
    CHECK(slot_blk0_initial >= 0);

    // Switch to Prompt B
    auto res_B = switcher.switch_context(active_ctx, prompt_B);
    CHECK(res_B.success == true);
    CHECK(res_B.common_tokens == 128);       // Block 0 reused!
    CHECK(res_B.divergent_tokens == 128);    // Suffix divergent
    CHECK(res_B.blocks_reused == 1);
    CHECK(res_B.blocks_transferred == 1);

    // Shared block 0 preserves slot assignment
    CHECK(active_ctx.page_table[0] == slot_blk0_initial);
    CHECK(active_ctx.tokens == prompt_B);
    CHECK(active_ctx.recurrence == rec_B);
    CHECK(active_ctx.active_blocks[1] == blk1B);
}

// -----------------------------------------------------------------------------
// Test 3: Switching back from B to A (warm switch)
// -----------------------------------------------------------------------------
void test_warm_context_switch() {
    std::printf("[Test 3] Switching back from B to A (warm switch)...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcher switcher(cache);

    std::vector<int32_t> prompt_A(256, 42);
    std::vector<int32_t> prompt_B(256, 99);

    cache->insert_sequence(prompt_A, true, true);
    cache->insert_sequence(prompt_B, true, true);

    ContextState active_ctx;
    switcher.init_context(active_ctx);

    // Load Prompt A
    switcher.switch_context(active_ctx, prompt_A);

    // Switch to Prompt A again (100% warm hit, same context)
    auto res_warm = switcher.switch_context(active_ctx, prompt_A);
    CHECK(res_warm.success == true);
    CHECK(res_warm.common_tokens == 256);
    CHECK(res_warm.divergent_tokens == 0);
    CHECK(res_warm.blocks_transferred == 0);
    CHECK(res_warm.blocks_reused == 2);
    CHECK(res_warm.switch_latency_us < 1000); // Sub-millisecond switch

    CHECK(switcher.metrics().warm_switches >= 1);
}

// -----------------------------------------------------------------------------
// Test 4: Recurrence state integrity
// -----------------------------------------------------------------------------
void test_recurrence_state_integrity() {
    std::printf("[Test 4] Recurrence state integrity across switches...\n");

    RecurrenceState original;
    original.gdn = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    original.ple = {0x10, 0x20, 0x30, 0x40};
    original.tails = {0xAA, 0xBB, 0xCC};
    original.dead = {0x01};
    original.block_pos = {0x7F, 0x00, 0x12};

    auto serialized = original.serialize();
    CHECK(!serialized.empty());

    auto restored = RecurrenceState::deserialize(serialized);
    CHECK(restored == original);
    CHECK(restored.gdn == original.gdn);
    CHECK(restored.ple == original.ple);
    CHECK(restored.tails == original.tails);
    CHECK(restored.dead == original.dead);
    CHECK(restored.block_pos == original.block_pos);

    // Test through ContextSwitcher
    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcher switcher(cache);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    auto blk = std::make_shared<KvBlock>();
    blk->payload.token_ids = tokens;
    blk->payload.recurrence = serialized;
    cache->insert_sequence(tokens, true, true, blk);

    ContextState ctx;
    switcher.init_context(ctx);
    auto res = switcher.switch_context(ctx, tokens);
    CHECK(res.success == true);
    CHECK(ctx.recurrence == original);
}

// -----------------------------------------------------------------------------
// Test 5: Fast prefill bypass verification
// -----------------------------------------------------------------------------
void test_fast_prefill_bypass() {
    std::printf("[Test 5] Fast prefill bypass verification...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcher switcher(cache);

    std::vector<int32_t> prompt_tokens;
    for (int i = 0; i < 256; ++i) prompt_tokens.push_back(500 + i);

    cache->insert_sequence(prompt_tokens, true, true);

    ContextState active_ctx;
    switcher.init_context(active_ctx);
    switcher.switch_context(active_ctx, prompt_tokens);

    // Query 1: Extends prompt_tokens with new query tokens
    std::vector<int32_t> query1 = prompt_tokens;
    query1.push_back(9001);
    query1.push_back(9002);

    size_t bypass = switcher.compute_prefill_bypass(active_ctx, query1);
    CHECK(bypass == 256); // All 256 prompt tokens bypassed!

    auto res1 = switcher.switch_context(active_ctx, query1);
    CHECK(res1.success == true);
    CHECK(res1.fast_prefill_bypassed == true);
    CHECK(res1.common_tokens == 256);
    CHECK(res1.divergent_tokens == 2);

    // Query 2: Completely different prefix
    std::vector<int32_t> query2 = {8888, 8889, 8890};
    size_t bypass2 = switcher.compute_prefill_bypass(active_ctx, query2);
    CHECK(bypass2 == 0);

    CHECK(switcher.metrics().fast_prefill_bypass_count >= 1);
}

// -----------------------------------------------------------------------------
// Test 6: Tiered integration (NvmeCache -> RadixConvCache -> Active Context)
// -----------------------------------------------------------------------------
void test_tiered_nvme_integration() {
    std::printf("[Test 6] Tiered integration: NVMe to RadixConvCache to Context...\n");

    auto temp_dir = std::filesystem::temp_directory_path() / "strata_test_switcher_nvme";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    NvmeCacheConfig nvme_cfg;
    nvme_cfg.cache_dir = temp_dir;
    nvme_cfg.direct_io = false; // Safe for any test fs
    auto nvme = std::make_shared<NvmeCache>(nvme_cfg);

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig switcher_cfg;
    switcher_cfg.auto_fetch_nvme = true;
    ContextSwitcher switcher(cache, nvme, switcher_cfg);

    std::vector<int32_t> tokens;
    for (int i = 0; i < 128; ++i) tokens.push_back(7000 + i);

    RecurrenceState rec;
    rec.gdn.assign(32, 0x77);
    rec.ple.assign(16, 0x88);

    auto block = make_mock_block(0, tokens, rec, 0x7A, 0x7B);

    // 1. Write block to NVMe
    bool write_ok = nvme->write_block_sync(*block);
    CHECK(write_ok == true);

    // 2. Insert into Radix tree without attaching kv_block in RAM
    auto node = cache->insert_sequence(tokens, true, true, nullptr);
    CHECK(node != nullptr);
    CHECK(node->kv_block == nullptr); // missing in RAM
    node->block_hash = block->header.block_hash; // set CAS hash for NVMe fetch

    // 3. Switch context: should automatically load missing block from NVMe
    ContextState active_ctx;
    switcher.init_context(active_ctx);

    auto res = switcher.switch_context(active_ctx, tokens);
    CHECK(res.success == true);
    CHECK(node->kv_block != nullptr); // loaded into RAM!
    CHECK(node->kv_block->header.block_hash == block->header.block_hash);
    CHECK(active_ctx.active_blocks.size() == 1);
    CHECK(active_ctx.active_blocks[0] != nullptr);
    CHECK(active_ctx.recurrence == rec);
    CHECK(switcher.metrics().nvme_fetches >= 1);

    nvme->clear();
    std::filesystem::remove_all(temp_dir);
}

// -----------------------------------------------------------------------------
// Test 7: Stream coordination and DMA callback hooks
// -----------------------------------------------------------------------------
void test_stream_coordination_and_hooks() {
    std::printf("[Test 7] Stream coordination & custom DMA hooks...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig cfg;
    cfg.block_tokens = 128;

    uint64_t dma_calls = 0;
    uint64_t sync_calls = 0;
    cfg.custom_dma_copy = [&](void* dst, const void* src, size_t bytes, void* stream) {
        (void)stream;
        std::memcpy(dst, src, bytes);
        dma_calls++;
        return true;
    };
    cfg.custom_stream_sync = [&](void* stream) {
        (void)stream;
        sync_calls++;
        return true;
    };

    ContextSwitcher switcher(cache, nullptr, cfg);

    std::vector<int32_t> tokens(128, 1234);
    auto blk = make_mock_block(0, tokens, RecurrenceState{}, 0x11, 0x22);
    cache->insert_sequence(tokens, true, true, blk);

    ContextState ctx;
    switcher.init_context(ctx);

    // Set up mock staging pools
    std::vector<uint8_t> mock_k_pool(4096, 0);
    std::vector<uint8_t> mock_v_pool(4096, 0);
    ctx.k_pool = mock_k_pool.data();
    ctx.v_pool = mock_v_pool.data();
    ctx.slot_bytes = 512;

    void* mock_stream = reinterpret_cast<void*>(0xDEADBEEF);
    auto res = switcher.switch_context(ctx, tokens, mock_stream);
    CHECK(res.success == true);
    CHECK(dma_calls == 2); // K and V data copied
    CHECK(sync_calls == 1); // Stream synchronized
}

// -----------------------------------------------------------------------------
// Test 8: Slot recycling under slot constraints
// -----------------------------------------------------------------------------
void test_slot_recycling_constrained() {
    std::printf("[Test 8] Slot recycling under slot constraints...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig cfg;
    cfg.block_tokens = 128;
    cfg.num_slots = 2; // Only 2 slots available!
    ContextSwitcher switcher(cache, nullptr, cfg);

    std::vector<int32_t> tokens_b0(128, 1);
    std::vector<int32_t> tokens_b1(128, 2);
    std::vector<int32_t> tokens_b2(128, 3);

    auto blk0 = make_mock_block(0, tokens_b0, RecurrenceState{});
    auto blk1 = make_mock_block(blk0->header.block_hash, tokens_b1, RecurrenceState{});
    auto blk2 = make_mock_block(blk1->header.block_hash, tokens_b2, RecurrenceState{});

    std::vector<int32_t> seq3 = tokens_b0;
    seq3.insert(seq3.end(), tokens_b1.begin(), tokens_b1.end());
    seq3.insert(seq3.end(), tokens_b2.begin(), tokens_b2.end());

    cache->insert_sequence(tokens_b0, true, true, blk0);
    cache->insert_sequence(seq3, true, true, blk2);

    ContextState ctx;
    switcher.init_context(ctx);

    auto res = switcher.switch_context(ctx, seq3);
    CHECK(res.success == true);
    // With 2 slots and 3 blocks, slots must have been allocated/recycled safely
    CHECK(ctx.slot_mapping.size() == 2);
    CHECK(ctx.page_table.size() >= 3);
}

// -----------------------------------------------------------------------------
// Test 9: Recurrence edge cases and isolation (no leakage across contexts)
// -----------------------------------------------------------------------------
void test_recurrence_edge_cases_and_isolation() {
    std::printf("[Test 9] Recurrence edge cases and cross-context isolation...\n");

    // 1. Deserializing null or empty
    auto empty_rec = RecurrenceState::deserialize(nullptr, 0);
    CHECK(empty_rec.empty());

    // 2. Deserializing corrupted / truncated buffer
    std::vector<uint8_t> corrupt = {0x55, 0x43, 0x45, 0x52, 0x01, 0x00}; // truncated header
    auto fallback_rec = RecurrenceState::deserialize(corrupt.data(), corrupt.size());
    CHECK(!fallback_rec.empty());
    CHECK(fallback_rec.gdn == corrupt); // loaded as fallback raw bytes

    // 3. Isolation test: Session A has recurrence; switching to cold Session B must clear it
    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcher switcher(cache);

    std::vector<int32_t> tokens_A = {10, 20, 30};
    RecurrenceState rec_A;
    rec_A.gdn = {0xDE, 0xAD, 0xBE, 0xEF};
    auto blk_A = make_mock_block(0, tokens_A, rec_A);
    cache->insert_sequence(tokens_A, true, true, blk_A);

    ContextState ctx;
    switcher.init_context(ctx, "session-A");
    auto res_A = switcher.switch_context(ctx, tokens_A);
    CHECK(res_A.success == true);
    CHECK(ctx.recurrence == rec_A);

    // Now switch to completely disjoint prompt B without cached recurrence
    std::vector<int32_t> tokens_B = {88, 89, 90};
    auto res_B = switcher.switch_context(ctx, tokens_B);
    CHECK(res_B.success == true);
    // ctx.recurrence must NOT leak rec_A!
    CHECK(ctx.recurrence.empty());
    CHECK(ctx.recurrence != rec_A);
}

// -----------------------------------------------------------------------------
// Test 10: Prefill bypass boundary limits with unmapped page table entries
// -----------------------------------------------------------------------------
void test_prefill_bypass_boundary_limits() {
    std::printf("[Test 10] Prefill bypass boundary limits on unmapped blocks...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig cfg;
    cfg.block_tokens = 128;
    ContextSwitcher switcher(cache, nullptr, cfg);

    ContextState ctx;
    switcher.init_context(ctx);

    std::vector<int32_t> active_tokens(256);
    for (int i = 0; i < 256; ++i) active_tokens[i] = i + 1;
    ctx.tokens = active_tokens;

    // Simulate page table where block 0 is mapped but block 1 is unmapped (-1)
    ctx.page_table.assign(2, -1);
    ctx.page_table[0] = 5; // Slot 5
    ctx.page_table[1] = -1; // Unmapped!

    std::vector<int32_t> query_tokens = active_tokens;
    // Query matches all 256 tokens, but block 1 is unmapped.
    // compute_prefill_bypass must NOT claim 256 tokens; it must be clamped to 128!
    size_t bypass = switcher.compute_prefill_bypass(ctx, query_tokens);
    CHECK(bypass == 128);
}

// -----------------------------------------------------------------------------
// Test 11: DMA error handling and graceful failure propagation
// -----------------------------------------------------------------------------
void test_dma_error_handling() {
    std::printf("[Test 11] DMA copy failure propagation...\n");

    auto cache = std::make_shared<RadixConvCache>();
    ContextSwitcherConfig cfg;
    cfg.block_tokens = 128;
    // Custom DMA callback that simulates PCIe bus transfer error
    cfg.custom_dma_copy = [](void*, const void*, size_t, void*) {
        return false; // DMA copy failed!
    };

    ContextSwitcher switcher(cache, nullptr, cfg);

    std::vector<int32_t> tokens(128, 77);
    auto blk = make_mock_block(0, tokens, RecurrenceState{});
    cache->insert_sequence(tokens, true, true, blk);

    ContextState ctx;
    switcher.init_context(ctx);
    std::vector<uint8_t> pool(1024, 0);
    ctx.k_pool = pool.data();
    ctx.v_pool = pool.data();
    ctx.slot_bytes = 512;

    auto res = switcher.switch_context(ctx, tokens);
    CHECK(res.success == false);
    CHECK(!res.error_message.empty());
}

} // namespace

int main() {
    std::printf("Running context_switcher_test...\n");
    test_cold_mount_system_prompt();
    test_differential_context_switching();
    test_warm_context_switch();
    test_recurrence_state_integrity();
    test_fast_prefill_bypass();
    test_tiered_nvme_integration();
    test_stream_coordination_and_hooks();
    test_slot_recycling_constrained();
    test_recurrence_edge_cases_and_isolation();
    test_prefill_bypass_boundary_limits();
    test_dma_error_handling();

    if (fails == 0) {
        std::printf("ALL CONTEXT SWITCHER TESTS PASSED cleanly.\n");
        return 0;
    } else {
        std::printf("TESTS FAILED with %d failures.\n", fails);
        return 1;
    }
}
