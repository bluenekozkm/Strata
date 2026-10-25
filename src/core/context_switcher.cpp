// src/core/context_switcher.cpp - PCIe DMA Context Switcher & Stream Link implementation.
#include "strata/core/context_switcher.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace strata::core {

namespace {

#pragma pack(push, 1)
struct RecurrenceHeader {
    uint32_t magic = 0x52454355; // 'RECU'
    uint32_t version = 1;
    uint32_t gdn_size = 0;
    uint32_t ple_size = 0;
    uint32_t tails_size = 0;
    uint32_t dead_size = 0;
    uint32_t block_pos_size = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)
static_assert(sizeof(RecurrenceHeader) == 32, "RecurrenceHeader must be 32 bytes");

inline uint64_t current_time_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

std::vector<uint8_t> RecurrenceState::serialize() const {
    RecurrenceHeader hdr;
    hdr.magic = 0x52454355;
    hdr.version = 1;
    hdr.gdn_size = static_cast<uint32_t>(gdn.size());
    hdr.ple_size = static_cast<uint32_t>(ple.size());
    hdr.tails_size = static_cast<uint32_t>(tails.size());
    hdr.dead_size = static_cast<uint32_t>(dead.size());
    hdr.block_pos_size = static_cast<uint32_t>(block_pos.size());
    hdr.reserved = 0;

    size_t total = sizeof(RecurrenceHeader) + gdn.size() + ple.size() + tails.size() + dead.size() + block_pos.size();
    std::vector<uint8_t> buf(total);
    std::memcpy(buf.data(), &hdr, sizeof(RecurrenceHeader));

    size_t off = sizeof(RecurrenceHeader);
    if (!gdn.empty()) {
        std::memcpy(buf.data() + off, gdn.data(), gdn.size());
        off += gdn.size();
    }
    if (!ple.empty()) {
        std::memcpy(buf.data() + off, ple.data(), ple.size());
        off += ple.size();
    }
    if (!tails.empty()) {
        std::memcpy(buf.data() + off, tails.data(), tails.size());
        off += tails.size();
    }
    if (!dead.empty()) {
        std::memcpy(buf.data() + off, dead.data(), dead.size());
        off += dead.size();
    }
    if (!block_pos.empty()) {
        std::memcpy(buf.data() + off, block_pos.data(), block_pos.size());
        off += block_pos.size();
    }
    return buf;
}

RecurrenceState RecurrenceState::deserialize(const uint8_t* data, size_t size) {
    RecurrenceState st;
    if (data == nullptr || size == 0) {
        return st;
    }
    if (size >= sizeof(RecurrenceHeader)) {
        RecurrenceHeader hdr;
        std::memcpy(&hdr, data, sizeof(RecurrenceHeader));
        if (hdr.magic == 0x52454355) {
            uint64_t req = sizeof(RecurrenceHeader) + uint64_t(hdr.gdn_size) + uint64_t(hdr.ple_size) +
                           uint64_t(hdr.tails_size) + uint64_t(hdr.dead_size) + uint64_t(hdr.block_pos_size);
            if (size >= req) {
                size_t off = sizeof(RecurrenceHeader);
                if (hdr.gdn_size > 0) {
                    st.gdn.assign(data + off, data + off + hdr.gdn_size);
                    off += hdr.gdn_size;
                }
                if (hdr.ple_size > 0) {
                    st.ple.assign(data + off, data + off + hdr.ple_size);
                    off += hdr.ple_size;
                }
                if (hdr.tails_size > 0) {
                    st.tails.assign(data + off, data + off + hdr.tails_size);
                    off += hdr.tails_size;
                }
                if (hdr.dead_size > 0) {
                    st.dead.assign(data + off, data + off + hdr.dead_size);
                    off += hdr.dead_size;
                }
                if (hdr.block_pos_size > 0) {
                    st.block_pos.assign(data + off, data + off + hdr.block_pos_size);
                    off += hdr.block_pos_size;
                }
                return st;
            }
        }
    }
    // Fallback: load raw unformatted bytes into gdn buffer
    st.gdn.assign(data, data + size);
    return st;
}

ContextSwitcher::ContextSwitcher()
    : ContextSwitcher(nullptr, nullptr, ContextSwitcherConfig{}) {}

ContextSwitcher::ContextSwitcher(std::shared_ptr<RadixConvCache> radix_cache,
                                 std::shared_ptr<platform::NvmeCache> nvme_cache,
                                 ContextSwitcherConfig config)
    : radix_cache_(std::move(radix_cache)),
      nvme_cache_(std::move(nvme_cache)),
      config_(std::move(config)) {}

ContextSwitcher::~ContextSwitcher() = default;

void ContextSwitcher::set_radix_cache(std::shared_ptr<RadixConvCache> radix_cache) {
    std::lock_guard<std::mutex> lock(mu_);
    radix_cache_ = std::move(radix_cache);
}

void ContextSwitcher::set_nvme_cache(std::shared_ptr<platform::NvmeCache> nvme_cache) {
    std::lock_guard<std::mutex> lock(mu_);
    nvme_cache_ = std::move(nvme_cache);
}

void ContextSwitcher::reset_metrics() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    metrics_.reset();
}

bool ContextSwitcher::init_context(ContextState& ctx, const std::string& session_id) const {
    ctx.session_id = session_id;
    ctx.tokens.clear();
    ctx.recurrence.clear();
    ctx.active_blocks.clear();
    ctx.active_node.reset();
    ctx.active_node_id = 0;
    ctx.last_switch_us = 0;

    size_t max_blocks = config_.max_blocks > 0 ? config_.max_blocks : 1024;
    size_t num_slots = config_.num_slots > 0 ? config_.num_slots : 512;

    ctx.page_table.assign(max_blocks, -1);
    ctx.slot_mapping.assign(num_slots, -1);
    return true;
}

size_t ContextSwitcher::compute_prefill_bypass(const ContextState& active_ctx,
                                              const std::vector<int32_t>& query_tokens) const {
    if (!config_.enable_fast_prefill_bypass || active_ctx.tokens.empty() || query_tokens.empty()) {
        return 0;
    }
    size_t max_cmp = std::min(active_ctx.tokens.size(), query_tokens.size());
    size_t common = 0;
    while (common < max_cmp && active_ctx.tokens[common] == query_tokens[common]) {
        ++common;
    }

    // Boundary check: ensure common prefix does not exceed contiguous mapped blocks
    if (!active_ctx.page_table.empty()) {
        size_t block_size = config_.block_tokens > 0 ? config_.block_tokens : kKvBlockTokens;
        size_t valid_blocks = 0;
        for (size_t b = 0; b < active_ctx.page_table.size(); ++b) {
            if (active_ctx.page_table[b] >= 0) {
                valid_blocks++;
            } else {
                break;
            }
        }
        size_t max_mapped_tokens = valid_blocks * block_size;
        if (common > max_mapped_tokens) {
            common = max_mapped_tokens;
        }
    }
    return common;
}

bool ContextSwitcher::perform_dma_copy(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0 || dst == nullptr || src == nullptr) {
        return true;
    }
    if (config_.custom_dma_copy) {
        return config_.custom_dma_copy(dst, src, bytes, stream);
    }
    if (stream == nullptr) {
        std::memcpy(dst, src, bytes);
        return true;
    }

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    using MemcpyAsyncFn = int (*)(void*, const void*, size_t, int, void*);
    static auto async_fn = []() -> MemcpyAsyncFn {
        auto fn = reinterpret_cast<MemcpyAsyncFn>(dlsym(RTLD_DEFAULT, "hipMemcpyAsync"));
        if (!fn) {
            fn = reinterpret_cast<MemcpyAsyncFn>(dlsym(RTLD_DEFAULT, "cudaMemcpyAsync"));
        }
        return fn;
    }();
    if (async_fn) {
        // kind: 1 = HostToDevice (in both HIP and CUDA runtimes)
        return async_fn(dst, src, bytes, 1, stream) == 0;
    }
#endif

    std::memcpy(dst, src, bytes);
    return true;
}

bool ContextSwitcher::synchronize_stream(void* stream) {
    if (stream == nullptr) {
        return true;
    }
    if (config_.custom_stream_sync) {
        return config_.custom_stream_sync(stream);
    }

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    using StreamSyncFn = int (*)(void*);
    static auto sync_fn = []() -> StreamSyncFn {
        auto fn = reinterpret_cast<StreamSyncFn>(dlsym(RTLD_DEFAULT, "hipStreamSynchronize"));
        if (!fn) {
            fn = reinterpret_cast<StreamSyncFn>(dlsym(RTLD_DEFAULT, "cudaStreamSynchronize"));
        }
        return fn;
    }();
    if (sync_fn) {
        return sync_fn(stream) == 0;
    }
#endif

    return true;
}

ContextSwitchResult ContextSwitcher::switch_context(ContextState& active_ctx,
                                                   const std::vector<int32_t>& target_tokens,
                                                   void* stream) {
    if (!radix_cache_) {
        return switch_internal(active_ctx, nullptr, target_tokens, {}, nullptr, stream);
    }
    auto match = radix_cache_->match_longest_prefix(target_tokens);
    std::shared_ptr<RadixKvNode> target_node = nullptr;
    if (match.is_exact && match.last_full_node &&
        match.last_full_node != radix_cache_->root() &&
        match.matched_tokens == target_tokens.size()) {
        target_node = match.last_full_node;
    } else {
        target_node = radix_cache_->insert_sequence(target_tokens);
    }
    return switch_internal(active_ctx, target_node, target_tokens, {}, nullptr, stream);
}

ContextSwitchResult ContextSwitcher::switch_context(ContextState& active_ctx,
                                                   const std::shared_ptr<RadixKvNode>& target_node,
                                                   void* stream) {
    if (!target_node) {
        ContextSwitchResult res;
        res.success = false;
        res.error_message = "target_node is null";
        return res;
    }
    std::vector<int32_t> tokens;
    if (radix_cache_) {
        tokens = radix_cache_->get_sequence_tokens(target_node);
    } else {
        std::vector<std::shared_ptr<RadixKvNode>> branch;
        auto curr = target_node;
        while (curr) {
            branch.push_back(curr);
            curr = curr->parent.lock();
        }
        std::reverse(branch.begin(), branch.end());
        for (const auto& n : branch) {
            tokens.insert(tokens.end(), n->tokens.begin(), n->tokens.end());
        }
    }
    return switch_internal(active_ctx, target_node, tokens, {}, nullptr, stream);
}

ContextSwitchResult ContextSwitcher::switch_context(ContextState& active_ctx,
                                                   uint64_t target_node_id,
                                                   void* stream) {
    if (!radix_cache_) {
        ContextSwitchResult res;
        res.success = false;
        res.error_message = "RadixConvCache is not configured";
        return res;
    }
    auto node = radix_cache_->find_node(target_node_id);
    if (!node) {
        ContextSwitchResult res;
        res.success = false;
        res.error_message = "target_node_id not found in RadixConvCache";
        return res;
    }
    return switch_context(active_ctx, node, stream);
}

ContextSwitchResult ContextSwitcher::switch_context(ContextState& active_ctx,
                                                   const ContextState& target_ctx,
                                                   void* stream) {
    return switch_internal(active_ctx, target_ctx.active_node, target_ctx.tokens,
                           target_ctx.active_blocks, &target_ctx.recurrence, stream);
}

ContextSwitchResult ContextSwitcher::switch_internal(
    ContextState& active_ctx,
    const std::shared_ptr<RadixKvNode>& target_node,
    const std::vector<int32_t>& target_tokens,
    const std::vector<std::shared_ptr<KvBlock>>& explicit_blocks,
    const RecurrenceState* explicit_recurrence,
    void* stream) {

    auto start_time = std::chrono::steady_clock::now();
    ContextSwitchResult result;

    if (active_ctx.page_table.empty() || active_ctx.slot_mapping.empty()) {
        init_context(active_ctx, active_ctx.session_id);
    }

    // 1. Gather target blocks
    std::vector<std::shared_ptr<KvBlock>> target_blocks;
    if (!explicit_blocks.empty()) {
        target_blocks = explicit_blocks;
    } else if (target_node && radix_cache_) {
        auto branch = radix_cache_->get_branch_nodes(target_node);
        for (auto& n : branch) {
            if (n->kv_block) {
                target_blocks.push_back(n->kv_block);
            } else if (config_.auto_fetch_nvme && nvme_cache_ && n->block_hash != 0) {
                KvBlock fetched;
                std::string err;
                if (nvme_cache_->read_block_sync(n->block_hash, fetched, &err)) {
                    auto blk = std::make_shared<KvBlock>(std::move(fetched));
                    radix_cache_->set_kv_block(n->node_id, blk);
                    target_blocks.push_back(blk);
                    std::lock_guard<std::mutex> lock(mu_);
                    metrics_.nvme_fetches++;
                }
            }
        }
    } else if (target_node) {
        if (target_node->kv_block) {
            target_blocks.push_back(target_node->kv_block);
        } else if (config_.auto_fetch_nvme && nvme_cache_ && target_node->block_hash != 0) {
            KvBlock fetched;
            std::string err;
            if (nvme_cache_->read_block_sync(target_node->block_hash, fetched, &err)) {
                auto blk = std::make_shared<KvBlock>(std::move(fetched));
                target_node->kv_block = blk;
                target_blocks.push_back(blk);
                std::lock_guard<std::mutex> lock(mu_);
                metrics_.nvme_fetches++;
            }
        }
    }

    // 2. Differential prefix token matching
    size_t common_tokens = 0;
    size_t max_cmp = std::min(active_ctx.tokens.size(), target_tokens.size());
    while (common_tokens < max_cmp && active_ctx.tokens[common_tokens] == target_tokens[common_tokens]) {
        ++common_tokens;
    }
    size_t divergent_tokens = target_tokens.size() - common_tokens;
    result.common_tokens = common_tokens;
    result.divergent_tokens = divergent_tokens;

    // 3. Block-level differential reuse
    size_t block_size = config_.block_tokens > 0 ? config_.block_tokens : kKvBlockTokens;
    size_t common_blocks = common_tokens / block_size;
    if (common_tokens == target_tokens.size() && !target_tokens.empty()) {
        common_blocks = (target_tokens.size() + block_size - 1) / block_size;
    }

    size_t active_b_count = active_ctx.active_blocks.size();
    size_t token_b_count = !target_tokens.empty() ? (target_tokens.size() + block_size - 1) / block_size : 0;
    size_t target_b_count = std::max(target_blocks.size(), token_b_count);

    size_t blocks_reused = std::min(common_blocks, std::min(active_b_count, target_b_count));
    result.blocks_reused = blocks_reused;

    size_t blocks_transferred = 0;
    size_t bytes_transferred = 0;

    // Unmap blocks that are beyond target_b_count
    for (size_t b = target_b_count; b < active_b_count; ++b) {
        if (b < active_ctx.page_table.size()) {
            int32_t slot = active_ctx.page_table[b];
            if (slot >= 0 && slot < static_cast<int32_t>(active_ctx.slot_mapping.size())) {
                active_ctx.slot_mapping[slot] = -1;
            }
            active_ctx.page_table[b] = -1;
        }
    }

    if (active_ctx.active_blocks.size() < target_b_count) {
        active_ctx.active_blocks.resize(target_b_count);
    }
    if (active_ctx.page_table.size() < target_b_count) {
        active_ctx.page_table.resize(target_b_count, -1);
    }

    // Transfer divergent blocks [blocks_reused, target_b_count)
    for (size_t b = blocks_reused; b < target_b_count; ++b) {
        int32_t slot = -1;
        if (b < active_ctx.page_table.size() && active_ctx.page_table[b] >= 0) {
            slot = active_ctx.page_table[b];
        } else {
            // Find free slot
            for (size_t s = 0; s < active_ctx.slot_mapping.size(); ++s) {
                if (active_ctx.slot_mapping[s] == -1) {
                    slot = static_cast<int32_t>(s);
                    break;
                }
            }
            if (slot == -1) {
                // Evict victim slot: prefer slot not used by target sequence [0, target_b_count)
                for (int32_t s = static_cast<int32_t>(active_ctx.slot_mapping.size()) - 1; s >= 0; --s) {
                    int32_t victim_b = active_ctx.slot_mapping[s];
                    if (victim_b >= static_cast<int32_t>(target_b_count) || victim_b < 0) {
                        if (victim_b >= 0 && victim_b < static_cast<int32_t>(active_ctx.page_table.size())) {
                            active_ctx.page_table[victim_b] = -1;
                        }
                        slot = s;
                        break;
                    }
                }
                if (slot == -1) {
                    // Fallback: evict any slot not currently holding an active block in [0, b)
                    for (int32_t s = static_cast<int32_t>(active_ctx.slot_mapping.size()) - 1; s >= 0; --s) {
                        int32_t victim_b = active_ctx.slot_mapping[s];
                        if (victim_b >= static_cast<int32_t>(b) || victim_b < 0) {
                            if (victim_b >= 0 && victim_b < static_cast<int32_t>(active_ctx.page_table.size())) {
                                active_ctx.page_table[victim_b] = -1;
                            }
                            slot = s;
                            break;
                        }
                    }
                }
                if (slot == -1 && !active_ctx.slot_mapping.empty()) {
                    slot = static_cast<int32_t>(active_ctx.slot_mapping.size()) - 1;
                    int32_t victim_b = active_ctx.slot_mapping[slot];
                    if (victim_b >= 0 && victim_b < static_cast<int32_t>(active_ctx.page_table.size())) {
                        active_ctx.page_table[victim_b] = -1;
                    }
                }
            }
        }

        if (slot >= 0) {
            if (b >= active_ctx.page_table.size()) {
                active_ctx.page_table.resize(b + 1, -1);
            }
            active_ctx.page_table[b] = slot;
            active_ctx.slot_mapping[slot] = static_cast<int32_t>(b);
        }

        std::shared_ptr<KvBlock> blk = (b < target_blocks.size()) ? target_blocks[b] : nullptr;
        if (blk) {
            active_ctx.active_blocks[b] = blk;
            size_t payload_bytes = blk->payload.total_payload_bytes();
            bytes_transferred += payload_bytes;

            if (active_ctx.k_pool && slot >= 0 && !blk->payload.k_data.empty()) {
                size_t stride = active_ctx.slot_bytes > 0 ? active_ctx.slot_bytes : blk->payload.k_data.size();
                uint8_t* dst_k = static_cast<uint8_t*>(active_ctx.k_pool) + slot * stride;
                if (!perform_dma_copy(dst_k, blk->payload.k_data.data(), blk->payload.k_data.size(), stream)) {
                    result.success = false;
                    result.error_message = "DMA copy failed for K block " + std::to_string(b);
                    return result;
                }
            }
            if (active_ctx.v_pool && slot >= 0 && !blk->payload.v_data.empty()) {
                size_t stride = active_ctx.slot_bytes > 0 ? active_ctx.slot_bytes : blk->payload.v_data.size();
                uint8_t* dst_v = static_cast<uint8_t*>(active_ctx.v_pool) + slot * stride;
                if (!perform_dma_copy(dst_v, blk->payload.v_data.data(), blk->payload.v_data.size(), stream)) {
                    result.success = false;
                    result.error_message = "DMA copy failed for V block " + std::to_string(b);
                    return result;
                }
            }
        } else {
            active_ctx.active_blocks[b] = nullptr;
        }

        blocks_transferred++;
    }

    if (active_ctx.active_blocks.size() > target_b_count) {
        active_ctx.active_blocks.resize(target_b_count);
    }

    result.blocks_transferred = blocks_transferred;
    result.bytes_transferred = bytes_transferred;

    // 4. Restore Recurrence State
    if (explicit_recurrence != nullptr) {
        active_ctx.recurrence = *explicit_recurrence;
    } else {
        bool found_rec = false;
        for (auto it = target_blocks.rbegin(); it != target_blocks.rend(); ++it) {
            if (*it && !(*it)->payload.recurrence.empty()) {
                active_ctx.recurrence = RecurrenceState::deserialize((*it)->payload.recurrence);
                found_rec = true;
                break;
            }
        }
        if (!found_rec && target_node && target_node->kv_block && !target_node->kv_block->payload.recurrence.empty()) {
            active_ctx.recurrence = RecurrenceState::deserialize(target_node->kv_block->payload.recurrence);
            found_rec = true;
        }
        if (!found_rec) {
            if (common_tokens < active_ctx.tokens.size() || active_ctx.tokens.empty()) {
                active_ctx.recurrence.clear();
            }
        }
    }

    // 5. Stream synchronization
    if (config_.sync_stream_on_switch && stream != nullptr) {
        if (!synchronize_stream(stream)) {
            result.success = false;
            result.error_message = "Stream synchronization failed";
            return result;
        }
    }

    // 6. Update Active Context metadata
    active_ctx.tokens = target_tokens;
    active_ctx.active_node = target_node;
    active_ctx.active_node_id = target_node ? target_node->node_id : 0;
    active_ctx.last_switch_us = current_time_us();

    // 7. Fast prefill bypass check
    if (config_.enable_fast_prefill_bypass && common_tokens > 0) {
        result.fast_prefill_bypassed = true;
    }

    auto end_time = std::chrono::steady_clock::now();
    uint64_t latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    result.switch_latency_us = latency_us;
    result.success = true;

    // 8. Update Metrics
    {
        std::lock_guard<std::mutex> lock(mu_);
        metrics_.total_switches++;
        metrics_.total_switch_latency_us += latency_us;
        metrics_.last_switch_latency_us = latency_us;
        metrics_.total_tokens_reused += common_tokens;
        metrics_.total_tokens_divergent += divergent_tokens;
        metrics_.total_dma_bytes += bytes_transferred;
        if (result.fast_prefill_bypassed) {
            metrics_.fast_prefill_bypass_count++;
        }
        if (common_tokens == target_tokens.size() && !target_tokens.empty()) {
            metrics_.warm_switches++;
        } else if (common_tokens > 0) {
            metrics_.partial_switches++;
        } else {
            metrics_.cold_switches++;
        }
    }

    return result;
}

} // namespace strata::core
