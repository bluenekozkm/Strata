// include/strata/core/context_switcher.hpp - PCIe DMA Context Switcher & Stream Link.
// Manages fast context swaps between conversational branches and System Prompts,
// differential prefix reuse, recurrence state restoration, and stream coordination.
#pragma once

#include "strata/core/kv_block.hpp"
#include "strata/core/radix_conv_cache.hpp"
#include "strata/platform/nvme_cache.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

/// Recurrence state for GDN, PLE history, and QSA indexer tails.
struct RecurrenceState {
    std::vector<uint8_t> gdn;        // Gated Delta Net recurrence & conv history
    std::vector<uint8_t> ple;        // Prompt lookup embedding history
    std::vector<uint8_t> tails;      // QSA indexer tail key snapshots
    std::vector<uint8_t> dead;       // Dead state flags
    std::vector<uint8_t> block_pos;  // Positional checkpoints

    size_t total_bytes() const noexcept {
        return gdn.size() + ple.size() + tails.size() + dead.size() + block_pos.size();
    }

    bool empty() const noexcept {
        return gdn.empty() && ple.empty() && tails.empty() && dead.empty() && block_pos.empty();
    }

    void clear() noexcept {
        gdn.clear();
        ple.clear();
        tails.clear();
        dead.clear();
        block_pos.clear();
    }

    bool operator==(const RecurrenceState& o) const = default;

    /// Serialize into a compact binary buffer (e.g. for KvBlockPayload::recurrence).
    std::vector<uint8_t> serialize() const;

    /// Deserialize from a compact binary buffer.
    static RecurrenceState deserialize(const uint8_t* data, size_t size);
    static RecurrenceState deserialize(const std::vector<uint8_t>& buf) {
        return deserialize(buf.data(), buf.size());
    }
};

/// Represents the active session context state on device and host.
struct ContextState {
    std::shared_ptr<RadixKvNode> active_node = nullptr;
    uint64_t active_node_id = 0;
    std::vector<int32_t> tokens;
    RecurrenceState recurrence;

    // Page table: logical block index -> physical slot index (-1 for unmapped)
    std::vector<int32_t> page_table;
    // Slot mapping: physical slot index -> logical block index (-1 for free)
    std::vector<int32_t> slot_mapping;

    // Mounted KV blocks for each logical block index
    std::vector<std::shared_ptr<KvBlock>> active_blocks;

    // Optional physical slot buffer pointers (mock or device UVA memory)
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    size_t slot_bytes = 0;

    std::string session_id;
    uint64_t last_switch_us = 0;

    size_t token_count() const noexcept { return tokens.size(); }
    size_t block_count() const noexcept { return active_blocks.size(); }
    bool is_valid() const noexcept { return active_node != nullptr || !tokens.empty(); }

    void reset() {
        active_node.reset();
        active_node_id = 0;
        tokens.clear();
        recurrence.clear();
        std::fill(page_table.begin(), page_table.end(), -1);
        std::fill(slot_mapping.begin(), slot_mapping.end(), -1);
        active_blocks.clear();
        session_id.clear();
        last_switch_us = 0;
    }
};

/// Callback hook types for custom DMA transfers and stream synchronization.
using DmaCopyFn = std::function<bool(void* dst, const void* src, size_t bytes, void* stream)>;
using StreamSyncFn = std::function<bool(void* stream)>;

/// Configuration for ContextSwitcher.
struct ContextSwitcherConfig {
    size_t max_blocks = 1024;               // Maximum logical blocks supported
    size_t num_slots = 512;                 // Physical VRAM/staging slots
    size_t block_tokens = kKvBlockTokens;   // 128 tokens per block
    int format = kKvInt8;                   // KV tensor format
    uint32_t num_layers = 1;
    uint32_t num_heads = 2;
    uint32_t head_dim = 256;
    bool enable_fast_prefill_bypass = true; // Skip prefill when common prefix is active
    bool auto_fetch_nvme = true;            // Fetch missing blocks from NvmeCache
    bool sync_stream_on_switch = true;      // Synchronize stream after DMA copies
    bool verbose = false;

    DmaCopyFn custom_dma_copy = nullptr;
    StreamSyncFn custom_stream_sync = nullptr;
};

/// Cumulative runtime metrics for ContextSwitcher.
struct ContextSwitchMetrics {
    uint64_t total_switches = 0;
    uint64_t warm_switches = 0;             // 100% tokens matched / zero transfer
    uint64_t partial_switches = 0;          // Prefix partially matched
    uint64_t cold_switches = 0;             // 0 tokens matched
    uint64_t total_tokens_reused = 0;
    uint64_t total_tokens_divergent = 0;
    uint64_t total_dma_bytes = 0;
    uint64_t fast_prefill_bypass_count = 0;
    uint64_t last_switch_latency_us = 0;
    uint64_t total_switch_latency_us = 0;
    uint64_t nvme_fetches = 0;

    double avg_latency_us() const noexcept {
        return total_switches > 0 ? static_cast<double>(total_switch_latency_us) / total_switches : 0.0;
    }
    double avg_latency_ms() const noexcept {
        return avg_latency_us() / 1000.0;
    }
    void reset() noexcept {
        *this = ContextSwitchMetrics{};
    }
    bool operator==(const ContextSwitchMetrics& o) const = default;
};

/// Result of a context switch operation.
struct ContextSwitchResult {
    bool success = false;
    size_t common_tokens = 0;
    size_t divergent_tokens = 0;
    size_t blocks_reused = 0;
    size_t blocks_transferred = 0;
    size_t bytes_transferred = 0;
    uint64_t switch_latency_us = 0;
    bool fast_prefill_bypassed = false;
    std::string error_message;
};

/// Coordinates PCIe DMA transfers and stream ordering between Radix Conv Cache and active session.
class ContextSwitcher {
public:
    ContextSwitcher();
    explicit ContextSwitcher(std::shared_ptr<RadixConvCache> radix_cache,
                             std::shared_ptr<platform::NvmeCache> nvme_cache = nullptr,
                             ContextSwitcherConfig config = {});
    ~ContextSwitcher();

    ContextSwitcher(const ContextSwitcher&) = delete;
    ContextSwitcher& operator=(const ContextSwitcher&) = delete;
    ContextSwitcher(ContextSwitcher&&) = delete;
    ContextSwitcher& operator=(ContextSwitcher&&) = delete;

    // ---- Context Switching Operations ----

    /// Switch active context to the target token sequence.
    ContextSwitchResult switch_context(ContextState& active_ctx,
                                      const std::vector<int32_t>& target_tokens,
                                      void* stream = nullptr);

    /// Switch active context to the target Radix node.
    ContextSwitchResult switch_context(ContextState& active_ctx,
                                      const std::shared_ptr<RadixKvNode>& target_node,
                                      void* stream = nullptr);

    /// Switch active context to target node by ID.
    ContextSwitchResult switch_context(ContextState& active_ctx,
                                      uint64_t target_node_id,
                                      void* stream = nullptr);

    /// Switch active context directly from another ContextState.
    ContextSwitchResult switch_context(ContextState& active_ctx,
                                      const ContextState& target_ctx,
                                      void* stream = nullptr);

    // ---- Session Context Lifecycle ----

    /// Initialize page table and slot mappings for a new or reset ContextState.
    bool init_context(ContextState& ctx, const std::string& session_id = "") const;

    /// Calculate how many leading tokens of query_tokens can bypass prompt prefill.
    size_t compute_prefill_bypass(const ContextState& active_ctx,
                                  const std::vector<int32_t>& query_tokens) const;

    // ---- Cache References & Accessors ----

    std::shared_ptr<RadixConvCache> radix_cache() const noexcept { return radix_cache_; }
    std::shared_ptr<platform::NvmeCache> nvme_cache() const noexcept { return nvme_cache_; }
    void set_radix_cache(std::shared_ptr<RadixConvCache> radix_cache);
    void set_nvme_cache(std::shared_ptr<platform::NvmeCache> nvme_cache);

    const ContextSwitchMetrics& metrics() const noexcept { return metrics_; }
    void reset_metrics() noexcept;

    const ContextSwitcherConfig& config() const noexcept { return config_; }

private:
    ContextSwitchResult switch_internal(ContextState& active_ctx,
                                       const std::shared_ptr<RadixKvNode>& target_node,
                                       const std::vector<int32_t>& target_tokens,
                                       const std::vector<std::shared_ptr<KvBlock>>& explicit_blocks,
                                       const RecurrenceState* explicit_recurrence,
                                       void* stream);

    bool perform_dma_copy(void* dst, const void* src, size_t bytes, void* stream);
    bool synchronize_stream(void* stream);

    std::shared_ptr<RadixConvCache> radix_cache_;
    std::shared_ptr<platform::NvmeCache> nvme_cache_;
    ContextSwitcherConfig config_;
    mutable std::mutex mu_;
    ContextSwitchMetrics metrics_;
};

} // namespace strata::core
