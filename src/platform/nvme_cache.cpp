// src/platform/nvme_cache.cpp - Async NVMe Storage Engine implementation.
#include "strata/platform/nvme_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <iomanip>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::platform {

namespace {

inline uint64_t now_us_u64() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

inline std::string hash_to_hex(uint64_t hash) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx.bin", static_cast<unsigned long long>(hash));
    return std::string(buf);
}

constexpr uint64_t kManifestSeed = 0x5354524D414E4946ULL; // 'STRMANIF'

} // namespace

NvmeCache::NvmeCache() : NvmeCache(NvmeCacheConfig{}) {}

NvmeCache::NvmeCache(const std::filesystem::path& cache_dir) {
    config_.cache_dir = cache_dir;
    init_directories();
    load_manifest();

    const size_t num_threads = std::max<size_t>(1, config_.num_worker_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&NvmeCache::worker_loop, this);
    }
}

NvmeCache::NvmeCache(NvmeCacheConfig config) : config_(std::move(config)) {
    init_directories();
    load_manifest();

    const size_t num_threads = std::max<size_t>(1, config_.num_worker_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&NvmeCache::worker_loop, this);
    }
}

NvmeCache::~NvmeCache() {
    flush_sync();

    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        stop_workers_ = true;
    }
    queue_cv_.notify_all();

    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    workers_.clear();

    save_manifest_atomic();
}

static std::atomic<uint64_t> s_file_tmp_seq{1};

void NvmeCache::init_directories() {
    std::error_code ec;
    std::filesystem::create_directories(config_.cache_dir, ec);
    std::filesystem::create_directories(config_.cache_dir / "blocks", ec);

    // Clean up any stale temporary files from previous aborted runs
    auto blocks_dir = config_.cache_dir / "blocks";
    if (std::filesystem::exists(blocks_dir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(blocks_dir, ec)) {
            if (entry.is_regular_file(ec)) {
                auto name = entry.path().filename().string();
                if (name.find(".tmp") != std::string::npos) {
                    std::filesystem::remove(entry.path(), ec);
                }
            }
        }
    }
}

std::filesystem::path NvmeCache::get_block_path(uint64_t block_hash) const {
    return config_.cache_dir / "blocks" / hash_to_hex(block_hash);
}

std::filesystem::path NvmeCache::manifest_path() const {
    return config_.cache_dir / ".strata_kv_manifest.bin";
}

bool NvmeCache::is_direct_io_active() const {
    return config_.direct_io && direct_io_supported_.load(std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Direct / Unbuffered File I/O
// -----------------------------------------------------------------------------

bool NvmeCache::write_file_direct(const std::filesystem::path& path,
                                  const uint8_t* data,
                                  size_t size,
                                  std::string& err) {
    if (!data || size == 0) {
        err = "Null or empty buffer";
        return false;
    }

    bool use_direct = is_direct_io_active() && (size % kKvBlockAlignment == 0) &&
                      ((reinterpret_cast<uintptr_t>(data) % kKvBlockAlignment) == 0);

    const auto tmp_path = path.string() + ".tmp." +
                          std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
                          "_" + std::to_string(s_file_tmp_seq.fetch_add(1, std::memory_order_relaxed));

#if defined(_WIN32)
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (use_direct) {
        flags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
    }

    HANDLE hFile = CreateFileW(tmp_path.c_str(), GENERIC_WRITE, 0, nullptr,
                               CREATE_ALWAYS, flags, nullptr);
    if (hFile == INVALID_HANDLE_VALUE && use_direct && config_.auto_direct_io_fallback) {
        direct_io_supported_.store(false, std::memory_order_relaxed);
        use_direct = false;
        flags = FILE_ATTRIBUTE_NORMAL;
        hFile = CreateFileW(tmp_path.c_str(), GENERIC_WRITE, 0, nullptr,
                            CREATE_ALWAYS, flags, nullptr);
    }

    if (hFile == INVALID_HANDLE_VALUE) {
        err = "CreateFile failed: " + std::to_string(GetLastError());
        return false;
    }

    size_t total_written = 0;
    while (total_written < size) {
        DWORD bytes_to_write = static_cast<DWORD>(std::min<size_t>(size - total_written, 64 * 1024 * 1024));
        DWORD written = 0;
        if (!WriteFile(hFile, data + total_written, bytes_to_write, &written, nullptr) || written == 0) {
            err = "WriteFile failed: " + std::to_string(GetLastError());
            CloseHandle(hFile);
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
        total_written += written;
    }

    FlushFileBuffers(hFile);
    CloseHandle(hFile);

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp_path, path, ec);
        if (ec) {
            err = "rename block file failed: " + ec.message();
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
    }
    return true;

#else // POSIX

    int flags = O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC;
    if (use_direct) {
#if defined(__APPLE__)
        // macOS does not have O_DIRECT; handled after open via fcntl F_NOCACHE
#else
        flags |= O_DIRECT;
#endif
    }

    int fd = ::open(tmp_path.c_str(), flags, 0644);
    if (fd < 0 && use_direct && config_.auto_direct_io_fallback &&
        (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOTTY)) {
        direct_io_supported_.store(false, std::memory_order_relaxed);
        use_direct = false;
        flags &= ~O_DIRECT;
        fd = ::open(tmp_path.c_str(), flags, 0644);
    }

    if (fd < 0) {
        err = "open failed: " + std::string(std::strerror(errno));
        return false;
    }

#if defined(__APPLE__)
    if (use_direct) {
        ::fcntl(fd, F_NOCACHE, 1);
    }
#endif

    size_t total_written = 0;
    while (total_written < size) {
        ssize_t n = ::pwrite(fd, data + total_written, size - total_written, static_cast<off_t>(total_written));
        if (n < 0) {
            if (errno == EINTR) continue;
            if (use_direct && errno == EINVAL && config_.auto_direct_io_fallback) {
                direct_io_supported_.store(false, std::memory_order_relaxed);
                ::close(fd);
                flags &= ~O_DIRECT;
                fd = ::open(tmp_path.c_str(), flags, 0644);
                if (fd < 0) {
                    err = "open fallback failed: " + std::string(std::strerror(errno));
                    std::error_code ec;
                    std::filesystem::remove(tmp_path, ec);
                    return false;
                }
                use_direct = false;
                total_written = 0;
                continue;
            }
            err = "pwrite failed: " + std::string(std::strerror(errno));
            ::close(fd);
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
        if (n == 0) break;
        total_written += static_cast<size_t>(n);
    }

    if (total_written < size) {
        err = "Short write to file: wrote " + std::to_string(total_written) + " of " + std::to_string(size);
        ::close(fd);
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return false;
    }

    ::fsync(fd);
    ::close(fd);

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        err = "rename block file failed: " + ec.message();
        std::filesystem::remove(tmp_path, ec);
        return false;
    }

    // Durably sync directory on POSIX
    int dfd = ::open((config_.cache_dir / "blocks").c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }

    return true;
#endif
}

bool NvmeCache::read_file_direct(const std::filesystem::path& path,
                                 AlignedBuffer& out_buf,
                                 size_t expected_size,
                                 std::string& err) {
    std::error_code ec;
    uintmax_t file_bytes = std::filesystem::file_size(path, ec);
    if (ec) {
        err = "Cannot stat file: " + ec.message();
        return false;
    }
    if (file_bytes == 0) {
        err = "Block file is empty (0 bytes)";
        return false;
    }
    if (expected_size > 0 && file_bytes < expected_size) {
        err = "Block file truncated: size " + std::to_string(file_bytes) +
              " < expected " + std::to_string(expected_size);
        return false;
    }

    const size_t bytes_to_read = (expected_size > 0) ? expected_size : file_bytes;
    const size_t alloc_bytes = strata::core::kv_align_up(bytes_to_read, kKvBlockAlignment);
    out_buf.resize(alloc_bytes, kKvBlockAlignment);

    bool use_direct = is_direct_io_active() && (bytes_to_read % kKvBlockAlignment == 0);

#if defined(_WIN32)
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (use_direct) {
        flags |= FILE_FLAG_NO_BUFFERING;
    }

    HANDLE hFile = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, flags, nullptr);
    if (hFile == INVALID_HANDLE_VALUE && use_direct && config_.auto_direct_io_fallback) {
        direct_io_supported_.store(false, std::memory_order_relaxed);
        use_direct = false;
        flags = FILE_ATTRIBUTE_NORMAL;
        hFile = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, flags, nullptr);
    }

    if (hFile == INVALID_HANDLE_VALUE) {
        err = "CreateFile for read failed: " + std::to_string(GetLastError());
        return false;
    }

    size_t total_read = 0;
    while (total_read < bytes_to_read) {
        DWORD want = static_cast<DWORD>(std::min<size_t>(bytes_to_read - total_read, 64 * 1024 * 1024));
        DWORD got = 0;
        if (!ReadFile(hFile, out_buf.data() + total_read, want, &got, nullptr)) {
            err = "ReadFile failed: " + std::to_string(GetLastError());
            CloseHandle(hFile);
            return false;
        }
        if (got == 0) break;
        total_read += got;
    }

    CloseHandle(hFile);
    if (total_read < bytes_to_read) {
        err = "Short read from file: got " + std::to_string(total_read) + " of " + std::to_string(bytes_to_read);
        return false;
    }
    return true;

#else // POSIX

    int flags = O_RDONLY | O_CLOEXEC;
    if (use_direct) {
#if defined(__APPLE__)
        // macOS
#else
        flags |= O_DIRECT;
#endif
    }

    int fd = ::open(path.c_str(), flags);
    if (fd < 0 && use_direct && config_.auto_direct_io_fallback &&
        (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOTTY)) {
        direct_io_supported_.store(false, std::memory_order_relaxed);
        use_direct = false;
        flags &= ~O_DIRECT;
        fd = ::open(path.c_str(), flags);
    }

    if (fd < 0) {
        err = "open for read failed: " + std::string(std::strerror(errno));
        return false;
    }

#if defined(__APPLE__)
    if (use_direct) {
        ::fcntl(fd, F_NOCACHE, 1);
    }
#endif

    size_t total_read = 0;
    while (total_read < bytes_to_read) {
        ssize_t n = ::pread(fd, out_buf.data() + total_read, bytes_to_read - total_read, static_cast<off_t>(total_read));
        if (n < 0) {
            if (errno == EINTR) continue;
            // If O_DIRECT failed with EINVAL (e.g. unaligned file size in corrupted file), retry without O_DIRECT
            if (use_direct && errno == EINVAL) {
                direct_io_supported_.store(false, std::memory_order_relaxed);
                ::close(fd);
                fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0) {
                    err = "open fallback failed: " + std::string(std::strerror(errno));
                    return false;
                }
                use_direct = false;
                total_read = 0;
                continue;
            }
            err = "pread failed: " + std::string(std::strerror(errno));
            ::close(fd);
            return false;
        }
        if (n == 0) break;
        total_read += static_cast<size_t>(n);
    }

    ::close(fd);
    if (total_read < bytes_to_read) {
        err = "Short read from file: got " + std::to_string(total_read) + " of " + std::to_string(bytes_to_read);
        return false;
    }
    return true;
#endif
}

// -----------------------------------------------------------------------------
// Manifest Management & Crash Consistency
// -----------------------------------------------------------------------------

void NvmeCache::load_manifest() {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    manifest_index_.clear();
    stats_ = NvmeCacheStats{};

    auto mpath = manifest_path();
    std::error_code ec;
    if (!std::filesystem::exists(mpath, ec)) {
        // Check if a previous crash occurred during rename (.tmp exists)
        auto tmp_path = mpath;
        tmp_path += ".tmp";
        if (std::filesystem::exists(tmp_path, ec)) {
            std::filesystem::rename(tmp_path, mpath, ec);
        }
    }

    if (!std::filesystem::exists(mpath, ec)) {
        return;
    }

    uintmax_t fsize = std::filesystem::file_size(mpath, ec);
    if (ec || fsize < sizeof(ManifestHeader)) {
        return;
    }

    FILE* fp = std::fopen(mpath.string().c_str(), "rb");
    if (!fp) return;

    ManifestHeader hdr;
    if (std::fread(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) {
        std::fclose(fp);
        return;
    }

    if (std::memcmp(hdr.magic, "STRMANIF", 8) != 0 || hdr.version != 1) {
        std::fclose(fp);
        return;
    }

    const size_t records_bytes = static_cast<size_t>(hdr.record_count) * sizeof(ManifestBlockRecord);
    if (fsize < sizeof(ManifestHeader) + records_bytes) {
        // Corrupted/truncated manifest
        std::fclose(fp);
        return;
    }

    std::vector<ManifestBlockRecord> records(hdr.record_count);
    if (records_bytes > 0) {
        if (std::fread(records.data(), 1, records_bytes, fp) != records_bytes) {
            std::fclose(fp);
            return;
        }

        uint64_t calc_checksum = strata::core::kv_block_checksum(records.data(), records_bytes, kManifestSeed);
        if (calc_checksum != hdr.checksum) {
            // Corrupted manifest
            std::fclose(fp);
            return;
        }
    }
    std::fclose(fp);

    for (const auto& rec : records) {
        if (!std::filesystem::exists(get_block_path(rec.block_hash), ec)) {
            // Block file missing on disk; mark dirty to update manifest on next sync
            manifest_dirty_ = true;
            continue;
        }
        manifest_index_[rec.block_hash] = rec;
        stats_.total_bytes += rec.file_size;
        stats_.block_count++;
        if (rec.is_pinned) {
            stats_.pinned_count++;
        }
    }
}

bool NvmeCache::save_manifest_atomic() {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    return save_manifest_atomic_locked();
}

bool NvmeCache::save_manifest_atomic_locked() {
    if (!manifest_dirty_) {
        return true;
    }

    ManifestHeader hdr;
    hdr.version = 1;
    hdr.record_count = static_cast<uint32_t>(manifest_index_.size());
    hdr.total_bytes = stats_.total_bytes;

    std::vector<ManifestBlockRecord> records;
    records.reserve(manifest_index_.size());
    for (const auto& [_, rec] : manifest_index_) {
        records.push_back(rec);
    }

    const size_t records_bytes = records.size() * sizeof(ManifestBlockRecord);
    if (records_bytes > 0) {
        hdr.checksum = strata::core::kv_block_checksum(records.data(), records_bytes, kManifestSeed);
    } else {
        hdr.checksum = 0;
    }

    const auto mpath = manifest_path();
    const auto tmp_path = mpath.string() + ".tmp";

    FILE* fp = std::fopen(tmp_path.c_str(), "wb");
    if (!fp) {
        return false;
    }

    if (std::fwrite(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) {
        std::fclose(fp);
        return false;
    }

    if (records_bytes > 0) {
        if (std::fwrite(records.data(), 1, records_bytes, fp) != records_bytes) {
            std::fclose(fp);
            return false;
        }
    }

    std::fflush(fp);

#if defined(_WIN32)
    int fd = _fileno(fp);
    if (fd >= 0) {
        HANDLE h = (HANDLE)_get_osfhandle(fd);
        if (h != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(h);
        }
    }
#else
    int fd = fileno(fp);
    if (fd >= 0) {
        ::fsync(fd);
    }
#endif

    std::fclose(fp);

    std::error_code ec;
    std::filesystem::rename(tmp_path, mpath, ec);
    if (ec) {
        std::filesystem::remove(mpath, ec);
        std::filesystem::rename(tmp_path, mpath, ec);
        if (ec) {
            return false;
        }
    }

#if !defined(_WIN32)
    int dfd = ::open(config_.cache_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
#endif

    manifest_dirty_ = false;
    return true;
}

// -----------------------------------------------------------------------------
// LRU Eviction Policy
// -----------------------------------------------------------------------------

size_t NvmeCache::evict_to_budget(size_t additional_bytes) {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    return evict_to_budget_locked(additional_bytes);
}

size_t NvmeCache::evict_to_budget_locked(size_t additional_bytes) {
    if (stats_.total_bytes + additional_bytes <= config_.disk_budget_bytes) {
        return 0;
    }

    struct Candidate {
        uint64_t block_hash;
        uint64_t last_access_us;
        uint64_t file_size;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(manifest_index_.size());

    for (const auto& [hash, rec] : manifest_index_) {
        if (!rec.is_pinned) {
            candidates.push_back({hash, rec.last_access_us, rec.file_size});
        }
    }

    // Sort oldest access first. Tie-break by hash for determinism.
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.last_access_us != b.last_access_us) {
            return a.last_access_us < b.last_access_us;
        }
        return a.block_hash < b.block_hash;
    });

    size_t evicted_count = 0;
    for (const auto& cand : candidates) {
        if (stats_.total_bytes + additional_bytes <= config_.disk_budget_bytes) {
            break;
        }

        auto path = get_block_path(cand.block_hash);
        std::error_code ec;
        std::filesystem::remove(path, ec);

        if (stats_.total_bytes >= cand.file_size) {
            stats_.total_bytes -= cand.file_size;
        } else {
            stats_.total_bytes = 0;
        }
        if (stats_.block_count > 0) {
            stats_.block_count--;
        }

        stats_.bytes_evicted += cand.file_size;
        stats_.evictions++;
        manifest_index_.erase(cand.block_hash);
        manifest_dirty_ = true;
        evicted_count++;
    }

    return evicted_count;
}

// -----------------------------------------------------------------------------
// Asynchronous & Synchronous Operations
// -----------------------------------------------------------------------------

void NvmeCache::write_block_async(const KvBlock& block, WriteCallback callback) {
    write_block_async(block, false, std::move(callback));
}

void NvmeCache::write_block_async(const KvBlock& block, bool is_pinned, WriteCallback callback) {
    auto buf = std::make_shared<AlignedBuffer>();
    auto err_code = strata::core::kv_block_serialize(block, *buf);
    if (err_code != strata::core::KvBlockError::kSuccess) {
        if (callback) {
            callback(false, strata::core::kv_block_error_string(err_code));
        }
        return;
    }

    uint64_t hash = block.header.block_hash;
    if (hash == 0) {
        hash = strata::core::kv_block_hash(block);
    }

    Task task;
    task.type = Task::Type::kWrite;
    task.block_hash = hash;
    task.write_buffer = buf;
    task.num_tokens = block.header.token_count;
    task.is_pinned = is_pinned;
    task.parent_hash = block.header.parent_hash;
    task.format = block.header.format;
    task.write_cb = std::move(callback);

    {
        std::lock_guard<std::mutex> lock(manifest_mu_);
        in_flight_writes_[hash] = InFlightEntry{buf, task.num_tokens, task.is_pinned, task.parent_hash, task.format};
    }

    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        if (stop_workers_) {
            {
                std::lock_guard<std::mutex> mlock(manifest_mu_);
                in_flight_writes_.erase(hash);
            }
            if (task.write_cb) task.write_cb(false, "NvmeCache is shutting down");
            return;
        }
        task_queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
}

void NvmeCache::write_buffer_async(uint64_t block_hash,
                                   AlignedBuffer buffer,
                                   uint32_t num_tokens,
                                   bool is_pinned,
                                   WriteCallback callback) {
    auto buf = std::make_shared<AlignedBuffer>(std::move(buffer));

    uint64_t parent_hash = 0;
    int32_t format = 0;
    if (buf->size() >= sizeof(KvBlockHeader)) {
        const auto* hdr = reinterpret_cast<const KvBlockHeader*>(buf->data());
        parent_hash = hdr->parent_hash;
        format = hdr->format;
        if (num_tokens == 0) {
            num_tokens = hdr->token_count;
        }
    }

    Task task;
    task.type = Task::Type::kWrite;
    task.block_hash = block_hash;
    task.write_buffer = buf;
    task.num_tokens = num_tokens;
    task.is_pinned = is_pinned;
    task.parent_hash = parent_hash;
    task.format = format;
    task.write_cb = std::move(callback);

    {
        std::lock_guard<std::mutex> lock(manifest_mu_);
        in_flight_writes_[block_hash] = InFlightEntry{buf, num_tokens, is_pinned, parent_hash, format};
    }

    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        if (stop_workers_) {
            {
                std::lock_guard<std::mutex> mlock(manifest_mu_);
                in_flight_writes_.erase(block_hash);
            }
            if (task.write_cb) task.write_cb(false, "NvmeCache is shutting down");
            return;
        }
        task_queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
}

void NvmeCache::read_block_async(uint64_t block_hash, ReadBlockCallback callback) {
    Task task;
    task.type = Task::Type::kReadBlock;
    task.block_hash = block_hash;
    task.read_block_cb = std::move(callback);

    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        if (stop_workers_) {
            if (task.read_block_cb) task.read_block_cb(false, KvBlock{}, "NvmeCache is shutting down");
            return;
        }
        task_queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
}

void NvmeCache::read_buffer_async(uint64_t block_hash, ReadBufferCallback callback) {
    Task task;
    task.type = Task::Type::kReadBuffer;
    task.block_hash = block_hash;
    task.read_buffer_cb = std::move(callback);

    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        if (stop_workers_) {
            if (task.read_buffer_cb) task.read_buffer_cb(false, AlignedBuffer{}, "NvmeCache is shutting down");
            return;
        }
        task_queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
}

bool NvmeCache::write_block_sync(const KvBlock& block, std::string* err) {
    return write_block_sync(block, false, err);
}

bool NvmeCache::write_block_sync(const KvBlock& block, bool is_pinned, std::string* err) {
    std::promise<std::pair<bool, std::string>> prom;
    auto fut = prom.get_future();

    write_block_async(block, is_pinned, [&prom](bool ok, const std::string& error) {
        prom.set_value({ok, error});
    });

    auto [ok, error_msg] = fut.get();
    if (err) *err = error_msg;
    return ok;
}

bool NvmeCache::write_buffer_sync(uint64_t block_hash,
                                  const AlignedBuffer& buffer,
                                  uint32_t num_tokens,
                                  bool is_pinned,
                                  std::string* err) {
    AlignedBuffer copy_buf(buffer.size(), kKvBlockAlignment);
    if (buffer.size() > 0 && copy_buf.data() && buffer.data()) {
        std::memcpy(copy_buf.data(), buffer.data(), buffer.size());
    }

    std::promise<std::pair<bool, std::string>> prom;
    auto fut = prom.get_future();

    write_buffer_async(block_hash, std::move(copy_buf), num_tokens, is_pinned,
                       [&prom](bool ok, const std::string& error) {
        prom.set_value({ok, error});
    });

    auto [ok, error_msg] = fut.get();
    if (err) *err = error_msg;
    return ok;
}

bool NvmeCache::read_block_sync(uint64_t block_hash, KvBlock& out_block, std::string* err) {
    std::promise<std::tuple<bool, KvBlock, std::string>> prom;
    auto fut = prom.get_future();

    read_block_async(block_hash, [&prom](bool ok, KvBlock block, const std::string& error) {
        prom.set_value({ok, std::move(block), error});
    });

    auto [ok, block, error_msg] = fut.get();
    if (ok) {
        out_block = std::move(block);
    }
    if (err) *err = error_msg;
    return ok;
}

bool NvmeCache::read_buffer_sync(uint64_t block_hash, AlignedBuffer& out_buffer, std::string* err) {
    std::promise<std::pair<bool, std::string>> prom;
    auto fut = prom.get_future();

    read_buffer_async(block_hash, [&prom, &out_buffer](bool ok, AlignedBuffer buf, const std::string& error) {
        if (ok) {
            out_buffer = std::move(buf);
        }
        prom.set_value({ok, error});
    });

    auto [ok, error_msg] = fut.get();
    if (err) *err = error_msg;
    return ok;
}

void NvmeCache::flush_sync() {
    {
        std::unique_lock<std::mutex> lock(queue_mu_);
        drain_cv_.wait(lock, [this]() {
            return task_queue_.empty() && active_workers_ == 0;
        });
    }

    save_manifest_atomic();
}

// -----------------------------------------------------------------------------
// Pinning & Metadata Queries
// -----------------------------------------------------------------------------

bool NvmeCache::pin_block(uint64_t block_hash, bool pinned) {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    auto it = manifest_index_.find(block_hash);
    if (it != manifest_index_.end()) {
        if (it->second.is_pinned != (pinned ? 1u : 0u)) {
            it->second.is_pinned = pinned ? 1u : 0u;
            if (pinned) {
                stats_.pinned_count++;
            } else if (stats_.pinned_count > 0) {
                stats_.pinned_count--;
            }
            manifest_dirty_ = true;
        }
        return true;
    }

    auto it_f = in_flight_writes_.find(block_hash);
    if (it_f != in_flight_writes_.end()) {
        it_f->second.is_pinned = pinned;
        return true;
    }

    return false;
}

bool NvmeCache::unpin_block(uint64_t block_hash) {
    return pin_block(block_hash, false);
}

bool NvmeCache::is_pinned(uint64_t block_hash) const {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    auto it = manifest_index_.find(block_hash);
    if (it != manifest_index_.end()) {
        return it->second.is_pinned != 0;
    }
    auto it_f = in_flight_writes_.find(block_hash);
    if (it_f != in_flight_writes_.end()) {
        return it_f->second.is_pinned;
    }
    return false;
}

bool NvmeCache::contains(uint64_t block_hash) const {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    return manifest_index_.contains(block_hash) || in_flight_writes_.contains(block_hash);
}

std::optional<ManifestBlockRecord> NvmeCache::get_metadata(uint64_t block_hash) const {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    auto it = manifest_index_.find(block_hash);
    if (it != manifest_index_.end()) {
        return it->second;
    }
    auto it_f = in_flight_writes_.find(block_hash);
    if (it_f != in_flight_writes_.end()) {
        ManifestBlockRecord rec;
        rec.block_hash = block_hash;
        rec.file_size = it_f->second.buffer ? it_f->second.buffer->size() : 0;
        rec.last_access_us = now_us_u64();
        rec.parent_hash = it_f->second.parent_hash;
        rec.num_tokens = it_f->second.num_tokens;
        rec.is_pinned = it_f->second.is_pinned ? 1u : 0u;
        rec.format = it_f->second.format;
        return rec;
    }
    return std::nullopt;
}

bool NvmeCache::remove_block(uint64_t block_hash) {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    in_flight_writes_.erase(block_hash);

    auto it = manifest_index_.find(block_hash);
    if (it == manifest_index_.end()) {
        return false;
    }

    auto path = get_block_path(block_hash);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (stats_.total_bytes >= it->second.file_size) {
        stats_.total_bytes -= it->second.file_size;
    } else {
        stats_.total_bytes = 0;
    }
    if (stats_.block_count > 0) {
        stats_.block_count--;
    }
    if (it->second.is_pinned && stats_.pinned_count > 0) {
        stats_.pinned_count--;
    }

    manifest_index_.erase(it);
    manifest_dirty_ = true;
    return true;
}

void NvmeCache::clear() {
    flush_sync();

    std::lock_guard<std::mutex> lock(manifest_mu_);
    in_flight_writes_.clear();
    for (const auto& [hash, _] : manifest_index_) {
        std::error_code ec;
        std::filesystem::remove(get_block_path(hash), ec);
    }
    manifest_index_.clear();
    stats_ = NvmeCacheStats{};
    manifest_dirty_ = true;
    save_manifest_atomic_locked();
}

NvmeCacheStats NvmeCache::get_stats() const {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    return stats_;
}

void NvmeCache::reset_stats() {
    std::lock_guard<std::mutex> lock(manifest_mu_);
    stats_.read_hits = 0;
    stats_.read_misses = 0;
    stats_.write_count = 0;
    stats_.write_errors = 0;
    stats_.evictions = 0;
    stats_.bytes_evicted = 0;
}

// -----------------------------------------------------------------------------
// Background Worker Loop & Task Execution
// -----------------------------------------------------------------------------

void NvmeCache::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_mu_);
            queue_cv_.wait(lock, [this]() {
                return stop_workers_ || !task_queue_.empty();
            });

            if (stop_workers_ && task_queue_.empty()) {
                break;
            }

            task = std::move(task_queue_.front());
            task_queue_.pop_front();
            ++active_workers_;
        }

        try {
            process_task(task);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[NvmeCache] Worker task exception: %s\n", ex.what());
            if (task.type == Task::Type::kWrite && task.write_cb) {
                try { task.write_cb(false, ex.what()); } catch (...) {}
            } else if (task.type == Task::Type::kReadBlock && task.read_block_cb) {
                try { task.read_block_cb(false, KvBlock{}, ex.what()); } catch (...) {}
            } else if (task.type == Task::Type::kReadBuffer && task.read_buffer_cb) {
                try { task.read_buffer_cb(false, AlignedBuffer{}, ex.what()); } catch (...) {}
            }
        } catch (...) {
            std::fprintf(stderr, "[NvmeCache] Worker task unknown exception\n");
        }

        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            --active_workers_;
            if (task_queue_.empty() && active_workers_ == 0) {
                drain_cv_.notify_all();
            }
        }
    }
}

void NvmeCache::process_task(Task& task) {
    if (task.type == Task::Type::kWrite) {
        const uint64_t hash = task.block_hash;
        const auto* buf = task.write_buffer.get();
        const size_t file_size = buf ? buf->size() : 0;
        const auto path = get_block_path(hash);

        // Check if block already safely exists on disk (CAS idempotency)
        bool already_stored = false;
        {
            std::lock_guard<std::mutex> lock(manifest_mu_);
            auto it = manifest_index_.find(hash);
            if (it != manifest_index_.end()) {
                std::error_code ec;
                if (std::filesystem::exists(path, ec)) {
                    already_stored = true;
                    if (task.is_pinned && !it->second.is_pinned) {
                        it->second.is_pinned = 1u;
                        stats_.pinned_count++;
                        manifest_dirty_ = true;
                    }
                    it->second.last_access_us = now_us_u64();
                    manifest_dirty_ = true;
                    in_flight_writes_.erase(hash);
                    stats_.write_count++;
                }
            }
        }

        if (already_stored) {
            if (config_.sync_manifest_on_write) {
                save_manifest_atomic();
            }
            if (task.write_cb) {
                try { task.write_cb(true, ""); } catch (...) {}
            }
            return;
        }

        // Pre-evict if budget exceeded
        {
            std::lock_guard<std::mutex> lock(manifest_mu_);
            evict_to_budget_locked(file_size);
        }

        std::string err;
        bool ok = false;
        if (buf && file_size > 0) {
            ok = write_file_direct(path, buf->data(), file_size, err);
        } else {
            err = "Empty write buffer";
        }

        {
            std::lock_guard<std::mutex> lock(manifest_mu_);
            in_flight_writes_.erase(hash);

            if (ok) {
                ManifestBlockRecord rec;
                rec.block_hash = hash;
                rec.file_size = file_size;
                rec.last_access_us = now_us_u64();
                rec.parent_hash = task.parent_hash;
                rec.num_tokens = task.num_tokens;
                rec.is_pinned = task.is_pinned ? 1u : 0u;
                rec.format = task.format;

                auto it = manifest_index_.find(hash);
                if (it != manifest_index_.end()) {
                    if (stats_.total_bytes >= it->second.file_size) {
                        stats_.total_bytes -= it->second.file_size;
                    }
                    if (it->second.is_pinned && stats_.pinned_count > 0) {
                        stats_.pinned_count--;
                    }
                    if (stats_.block_count > 0) {
                        stats_.block_count--;
                    }
                }

                manifest_index_[hash] = rec;
                stats_.total_bytes += file_size;
                stats_.block_count++;
                if (rec.is_pinned) {
                    stats_.pinned_count++;
                }
                stats_.write_count++;
                manifest_dirty_ = true;
            } else {
                stats_.write_errors++;
            }
        }

        if (config_.sync_manifest_on_write && ok) {
            save_manifest_atomic();
        }

        if (task.write_cb) {
            try { task.write_cb(ok, err); } catch (...) {}
        }

    } else if (task.type == Task::Type::kReadBlock || task.type == Task::Type::kReadBuffer) {
        const uint64_t hash = task.block_hash;
        AlignedBuffer read_buf;
        bool found = false;
        std::string err;

        // Check if block is in flight
        {
            std::lock_guard<std::mutex> lock(manifest_mu_);
            auto it_in_flight = in_flight_writes_.find(hash);
            if (it_in_flight != in_flight_writes_.end()) {
                const auto& entry = it_in_flight->second;
                if (entry.buffer && entry.buffer->size() > 0) {
                    read_buf.resize(entry.buffer->size(), kKvBlockAlignment);
                    if (read_buf.data() && entry.buffer->data()) {
                        std::memcpy(read_buf.data(), entry.buffer->data(), entry.buffer->size());
                        found = true;
                    }
                }
            }
        }

        if (!found) {
            size_t expected_size = 0;
            {
                std::lock_guard<std::mutex> lock(manifest_mu_);
                auto it = manifest_index_.find(hash);
                if (it != manifest_index_.end()) {
                    expected_size = it->second.file_size;
                }
            }

            const auto path = get_block_path(hash);
            found = read_file_direct(path, read_buf, expected_size, err);
        }

        if (!found) {
            {
                std::lock_guard<std::mutex> lock(manifest_mu_);
                stats_.read_misses++;
            }
            if (err.empty()) {
                err = "Block not found in cache: " + hash_to_hex(hash);
            }
            if (task.type == Task::Type::kReadBlock && task.read_block_cb) {
                try { task.read_block_cb(false, KvBlock{}, err); } catch (...) {}
            } else if (task.type == Task::Type::kReadBuffer && task.read_buffer_cb) {
                try { task.read_buffer_cb(false, AlignedBuffer{}, err); } catch (...) {}
            }
            return;
        }

        // Hit: update last_access_us and stats
        {
            std::lock_guard<std::mutex> lock(manifest_mu_);
            stats_.read_hits++;
            auto it = manifest_index_.find(hash);
            if (it != manifest_index_.end()) {
                it->second.last_access_us = now_us_u64();
                manifest_dirty_ = true;
            }
        }

        if (task.type == Task::Type::kReadBuffer) {
            if (task.read_buffer_cb) {
                try { task.read_buffer_cb(true, std::move(read_buf), ""); } catch (...) {}
            }
        } else {
            KvBlock block;
            auto deser_err = strata::core::kv_block_deserialize(read_buf, block, false);
            if (deser_err != strata::core::KvBlockError::kSuccess) {
                {
                    std::lock_guard<std::mutex> lock(manifest_mu_);
                    stats_.read_hits--;
                    stats_.read_misses++;
                }
                std::string err_str = "Deserialization failed: " +
                                      std::string(strata::core::kv_block_error_string(deser_err));
                if (task.read_block_cb) {
                    try { task.read_block_cb(false, KvBlock{}, err_str); } catch (...) {}
                }
            } else {
                if (task.read_block_cb) {
                    try { task.read_block_cb(true, std::move(block), ""); } catch (...) {}
                }
            }
        }
    }
}

} // namespace strata::platform
