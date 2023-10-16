// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, 
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include <assert.h>
#include "gpu-sim.h"
#include "hashing.h"
#include "stat-tool.h"

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels

const char *cache_request_status_str(enum cache_request_status status) {
  static const char *static_cache_request_status_str[] = {
      "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
      "SECTOR_MISS", "MSHR_HIT"};

  assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS);
  assert(status < NUM_CACHE_REQUEST_STATUS);

  return static_cache_request_status_str[status];
}

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  static const char *static_cache_reservation_fail_reason_str[] = {
      "LINE_ALLOC_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
      "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING"};

  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS);
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

  return static_cache_reservation_fail_reason_str[status];
}

/*

*/
unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
  // For sector cache, we select one sector per bank (sector interleaving)
  // This is what was found in Volta (one sector per bank, sector interleaving)
  // otherwise, line interleaving
  //对于扇区缓存，我们为每个存储体选择一个扇区（扇区交错）。
  //这是在Volta中发现的（每个存储体一个扇区，扇区交错），否则，行交错。
  return cache_config::hash_function(addr, l1_banks,
                                     l1_banks_byte_interleaving_log2,
                                     l1_banks_log2, l1_banks_hashing_function);
}

/*
返回一个地址在Cache中的set。
*/
unsigned cache_config::set_index(new_addr_type addr) const {
  // m_line_sz_log2 = LOGB2(m_line_sz);
  // m_nset_log2 = LOGB2(m_nset);
  // m_set_index_function = L1D是"L"-LINEAR_SET_FUNCTION，L2D是"P"-HASH_IPOLY_FUNCTION。
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function);
}

/*
返回一个地址在Cache中的set。
m_line_sz_log2 = LOGB2(m_line_sz);
m_nset_log2 = LOGB2(m_nset);
m_set_index_function = L1D是"L"-LINEAR_SET_FUNCTION，L2D是"P"-HASH_IPOLY_FUNCTION。
*/
unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
  unsigned set_index = 0;

  switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
      /*
       * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
       * Distance Theory" Cedric Nugteren et al. HPCA 2014
       */
      unsigned lower_xor = 0;
      unsigned upper_xor = 0;

      if (m_nset == 32 || m_nset == 64) {
        // Lower xor value is bits 7-11
        lower_xor = (addr >> m_line_sz_log2) & 0x1F;

        // Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        upper_xor |= (addr & 0x80000) >> 15;  // Bit 19

        set_index = (lower_xor ^ upper_xor);

        // 48KB cache prepends the set_index with bit 12
        if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;

      } else { /* Else incorrect number of sets for the hashing function */
        assert(
            "\nGPGPU-Sim cache configuration error: The number of sets should "
            "be "
            "32 or 64 for the hashing set index function.\n" &&
            0);
      }
      break;
    }

    case BITWISE_XORING_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = bitwise_hash_function(higher_bits, index, m_nset);
      break;
    }

    // V100配置的L2D Cache。
    case HASH_IPOLY_FUNCTION: {
      // addr: [m_line_sz_log2+m_nset_log2-1:0]                => set index + byte offset
      // addr: [:m_line_sz_log2+m_nset_log2]                   => Tag
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = ipoly_hash_function(higher_bits, index, m_nset);
      break;
    }
    case CUSTOM_SET_FUNCTION: {
      /* No custom set function implemented */
      break;
    }

    // V100配置的L1D Cache。
    case LINEAR_SET_FUNCTION: {
      // addr: [m_line_sz_log2-1:0]                            => byte offset
      // addr: [m_line_sz_log2+m_nset_log2-1:m_line_sz_log2]   => set index
      set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
      break;
    }

    default: {
      assert("\nUndefined set index function.\n" && 0);
      break;
    }
  }

  // Linear function selected or custom set index function not implemented
  assert((set_index < m_nset) &&
         "\nError: Set index out of bounds. This is caused by "
         "an incorrect or unimplemented custom set index function.\n");

  return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(m_config_string, FuncCachePreferNone);
  m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
  }

  return cache_config::set_index(part_addr);
}

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) {
  init(core_id, type_id);
}

void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  unsigned cache_lines_num = config.get_max_num_lines();
  m_lines = new cache_block_t *[cache_lines_num];
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  // initialize snapshot counters for visualizer
  m_prev_snapshot_access = 0;
  m_prev_snapshot_miss = 0;
  m_prev_snapshot_pending_hit = 0;
  m_core_id = core_id;
  m_type_id = type_id;
  is_used = false;
  m_dirty = 0;
}

void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

/*
判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS
/RESERVATION_FAIL等状态。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  return probe(addr, idx, mask, is_write, probe_mode, mf);
}

/*
判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS
/RESERVATION_FAIL等状态。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
  // assert( m_config.m_write_policy == READ_ONLY );
  //返回一个地址addr在Cache中的set index。
  unsigned set_index = m_config.set_index(addr);
  //为了便于起见，这里的标记包括index和Tag。这允许更复杂的（可能导致不同的indexes映射到
  //同一set）set index计算，因此需要完整的标签 + 索引来检查命中/未命中。Tag现在与块地址
  //相同。
  //这里实际返回的是除offset位以外的所有位，即set index也作为tag的一部分了。
  new_addr_type tag = m_config.tag(addr);

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;

  bool all_reserved = true;

  // check for hit or pending hit
  //对所有的Cache Ways检查。
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    // For example, 4 sets, 6 ways:
    // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
    // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
    // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
    // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
    //                |--------> index => cache_block_t *line
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    // Tag相符。
    if (line->m_tag == tag) {
      if (line->get_status(mask) == RESERVED) {
        //如果Cache block[mask]状态是RESERVED，说明有其他的线程正在读取这个Cache block。
        //挂起的命中访问已命中处于RESERVED状态的缓存行，这意味着同一行上已存在由先前缓存未
        //命中发送的flying内存请求。
        idx = index;
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        //如果Cache block[mask]状态是VALID，说明已经命中。
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        //如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当前访问也是写
        //操作的话即为命中，但如果不是写操作则需要判断是否mask标志的块是否修改完毕，修改完毕
        //则为命中，修改不完成则为SECTOR_MISS。因为L1 cache与L2 cache写命中时，采用write-
        //back策略，只将数据写入该block，并不直接更新下级存储，只有当这个块被替换时，才将数
        //据写回下级存储。
        //is_readable(mask)是判断mask标志的sector是否已经全部写完成，因为在修改cache的过程
        //中，有一个sector被修改即算作当前cache块MODIFIED，但是修改过程可能不是一下就能写完，
        //因此需要判断一下是否全部写完才可以算作读命中。
        if ((!is_write && line->is_readable(mask)) || is_write) {
          idx = index;
          return HIT;
        } else {
          idx = index;
          return SECTOR_MISS;
        }
      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        //Cache block有效，但是其中的byte mask=Cache block[mask]状态无效，说明sector缺失。
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
    //如果当前cache block的状态不是RESERVED。
    //到当前阶段，抛开前面能够确定的HIT，HIT_RESERVED，SECTOR_MISS，还剩下MISS/RESERVATION
    //_FAIL/MSHR_HIT三种状态。也就是说，在没有Hit，也没有HIT_RESERVED，也没有SECTOR_MISS的
    //情况下，需要逐出一个块，来给新访问提供RESERVE的空间。
    //line->is_reserved_line()：只要有一个sector是RESERVED，就认为这个cache block是RESERVED。
    if (!line->is_reserved_line()) {
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      // If the cacheline is from a load op (not modified), 
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement candidate
      // i.e. Only evict clean cachelines until total dirty cachelines reach the limit.
      //m_config.m_wr_percent在V100中配置为25%。
      //line->is_modified_line()：只要有一个sector是MODIFIED，就认为这个cache block是MODIFIED。
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) 
      {
        //一个cache block的状态有：INVALID = 0, RESERVED, VALID, MODIFIED，如果它是VALID，
        //就在上面的代码命中了
        //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
        //被MODIFIED，来逐出；但是如果dirty的cache block的比例超过m_wr_percent（V100中配置为
        //25%），也可以不满足MODIFIED的条件。
        
        //all_reserved被初始化为true，是指所有cache block都没有能够逐出来为新访问提供RESERVE
        //的空间，这里一旦满足上面两个if条件，说明当前line可以被逐出来提供空间供RESERVE新访问，
        //这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明当前line
        //不可被逐出，发生RESERVATION_FAIL。
        all_reserved = false;
        //line->is_invalid_line()是所有sector都无效。
        if (line->is_invalid_line()) {
          invalid_line = index;
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            //valid_timestamp设置为最近最少被使用的cache block的最末次访问时间。
            if (line->get_last_access_time() < valid_timestamp) {
              valid_timestamp = line->get_last_access_time();
              valid_line = index;
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            }
          }
        }
      }
    }
  }
  //Cache访问的状态包含：
  //    HIT，HIT_RESERVED，MISS，RESERVATION_FAIL，SECTOR_MISS，MSHR_HIT六种状态。
  //抛开前面能够确定的HIT，HIT_RESERVED，SECTOR_MISS还能够判断MISS/RESERVATION_FAIL
  //两种状态是否成立。
  //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
  //被MODIFIED，来逐出；但是如果dirty的cache block的比例超过m_wr_percent（V100中配置为
  //25%），也可以不满足MODIFIED的条件。
  //all_reserved被初始化为true，是指所有cache block都没有能够逐出来为新访问提供RESERVE
  //的空间，这里一旦满足上面两个if条件，说明cache block可以被逐出来提供空间供RESERVE新访
  //问，这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明cache
  //line不可被逐出，发生RESERVATION_FAIL。
  if (all_reserved) {
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  if (invalid_line != (unsigned)-1) {
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    idx = valid_line;
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable

  //if (probe_mode && m_config.is_streaming()) {
  //  line_table::const_iterator i =
  //      pending_lines.find(m_config.block_addr(addr));
  //  assert(mf);
  //  if (!mf->is_write() && i != pending_lines.end()) {
  //    if (i->second != mf->get_inst().get_uid()) return SECTOR_MISS;
  //  }
  //}

  //如果上面的cache block可以被逐出来reserve新访问，则返回MISS。
  return MISS;
}


/*
更新LRU状态。Least Recently Used。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;
  evicted_block_info evicted;
  enum cache_request_status result = access(addr, time, idx, wb, evicted, mf);
  assert(!wb);
  return result;
}

/*
更新LRU状态。Least Recently Used。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  m_access++;
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  //由于当前函数没有把之前probe函数的cache访问状态传参进来，这里这个probe单纯的重新获取这个状态。
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write());
  switch (status) {
    //新访问是HIT_RESERVED的话，不执行动作。
    case HIT_RESERVED:
      m_pending_hit++;
    //新访问是HIT的话，设置第idx号cache block以及mask对应的sector的最末此访问时间为当前拍。
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    //新访问是MISS的话，说明已经选定m_lines[idx]作为逐出并reserve新访问的cache block。
    case MISS:
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      //L1 cache与L2 cache均为allocate on miss。
      if (m_config.m_alloc_policy == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          //m_lines[idx]作为逐出并reserve新访问的cache block，如果它的某个sector已经被
          //MODIFIED，则需要执行写回操作，设置写回的标志为wb=true，设置逐出cache block的
          //信息。
          wb = true;
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask());
          //由于执行写回操作，MODIFIED造成的m_dirty数量应该减1。
          m_dirty--;
        }
        //执行对新访问的reserve操作。
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
      }
      break;
    //Cache block有效，但是其中的byte mask=Cache block[mask]状态无效，说明sector缺失。
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      //L1 cache与L2 cache均为allocate on miss。
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        //设置m_lines[idx]为新访问分配一个sector。
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    //probe函数中：
    //all_reserved被初始化为true，是指所有cache block都没有能够逐出来为新访问提供RESERVE
    //的空间，这里一旦满足函数两个if条件，说明cache block可以被逐出来提供空间供RESERVE新访
    //问，这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明cache
    //line不可被逐出，发生RESERVATION_FAIL。因此这里不执行任何操作。
    case RESERVATION_FAIL:
      m_res_fail++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
  fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(),
       is_write);
}

//
void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx;
  enum cache_request_status status = probe(addr, idx, mask, is_write);
  if (status == RESERVATION_FAIL) {
	  return;
  }
  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time,
                           mask);
  } else if (status == SECTOR_MISS) {
    assert(m_config.m_cache_type == SECTOR);
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
  }
  if (before && !m_lines[idx]->is_modified_line()) {
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line();
  m_lines[idx]->fill(time, mask, byte_mask);
  if (m_lines[idx]->is_modified_line() && !before) {
    m_dirty++;
  }
}

void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
  assert(m_config.m_alloc_policy == ON_MISS);
  bool before = m_lines[index]->is_modified_line();
  m_lines[index]->fill(time, mf->get_access_sector_mask(), mf->get_access_byte_mask());
  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));
      }
    }

  m_dirty = 0;
  is_used = false;
}

void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

  m_dirty = 0;
  is_used = false;
}

float tag_array::windowed_miss_rate() const {
  unsigned n_access = m_access - m_prev_snapshot_access;
  unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
  // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

  float missrate = 0.0f;
  if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
  return missrate;
}

void tag_array::new_window() {
  m_prev_snapshot_access = m_access;
  m_prev_snapshot_miss = m_miss;
  m_prev_snapshot_miss = m_miss + m_sector_miss;
  m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream);
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access);
  total_misses += (m_miss + m_sector_miss);
  total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;
  total_misses = (m_miss + m_sector_miss);
  total_hit_res = m_pending_hit;
  total_res_fail = m_res_fail;
}

/*
判断一系列的访问cache事件是否存在WRITE_REQUEST_SENT。
缓存事件类型包括：
    enum cache_event_type {
      //写回请求。
      WRITE_BACK_REQUEST_SENT,
      //读请求。
      READ_REQUEST_SENT,
      //写请求。
      WRITE_REQUEST_SENT,
      //写分配请求。
      WRITE_ALLOCATE_SENT
    };
*/
bool was_write_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_REQUEST_SENT) return true;
  }
  return false;
}

/*
判断一系列的访问cache事件是否存在WRITE_BACK_REQUEST_SENT。
缓存事件类型包括：
    enum cache_event_type {
      //写回请求。
      WRITE_BACK_REQUEST_SENT,
      //读请求。
      READ_REQUEST_SENT,
      //写请求。
      WRITE_REQUEST_SENT,
      //写分配请求。
      WRITE_ALLOCATE_SENT
    };
*/
bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
      wb_event = *e;
      return true;
    }
  }
  return false;
}

/*
判断一系列的访问cache事件是否存在READ_REQUEST_SENT。
缓存事件类型包括：
    enum cache_event_type {
      //写回请求。
      WRITE_BACK_REQUEST_SENT,
      //读请求。
      READ_REQUEST_SENT,
      //写请求。
      WRITE_REQUEST_SENT,
      //写分配请求。
      WRITE_ALLOCATE_SENT
    };
*/
bool was_read_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == READ_REQUEST_SENT) return true;
  }
  return false;
}

/*
判断一系列的访问cache事件是否存在WRITE_ALLOCATE_SENT。
缓存事件类型包括：
    enum cache_event_type {
      //写回请求。
      WRITE_BACK_REQUEST_SENT,
      //读请求。
      READ_REQUEST_SENT,
      //写请求。
      WRITE_REQUEST_SENT,
      //写分配请求。
      WRITE_ALLOCATE_SENT
    };
*/
bool was_writeallocate_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT) return true;
  }
  return false;
}
/****************************************************************** MSHR
 * ******************************************************************/

/*
Checks if there is a pending request to the lower memory level already.
检查是否已存在对较低内存级别的挂起请求。
*/
bool mshr_table::probe(new_addr_type block_addr) const {
  //MSHR表中的数据为std::unordered_map，是<new_addr_type, mshr_entry>的无序map。地址block_addr
  //去查找他是否在表中，如果 a = m_data.end()，则说明表中没有 block_addr；反之，则存在该条目。如果
  //不存在该条目，则返回false；如果存在该条目，返回true，代表存在对较低内存级别的挂起请求。
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/*
Checks if there is space for tracking a new memory access.
检查是否有空间处理新的内存访问。
*/
bool mshr_table::full(new_addr_type block_addr) const {
  //首先查找是否MSHR表中有 block_addr 地址的条目。
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    //如果存在该条目，看是否有空间合并进该条目。
    return i->second.m_list.size() >= m_max_merged;
  else
    //如果不存在该条目，看是否有其他空闲条目添加。
    return m_data.size() >= m_num_entries;
}

/*
Add or merge this access.
添加或合并此访问。这里假设的是MSHR表中有 block_addr 地址的条目，直接向该条目中添加。
*/
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  //将 block_addr 地址加入到对应条目内。
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR entry contains an atomic operation.
  //指示此MSHR条目包含原子操作。
  if (mf->isatomic()) {
    //mem_fetch定义了一个模拟内存请求的通信结构。更像是一个内存请求的行为。如果 mf 代表的内存访问是
    //原子操作，设置原子操作标志位。
    m_data[block_addr].m_has_atomic = true;
  }
}

/*
check is_read_after_write_pending.
检查是否存在挂起的写后读请求。这里假设的是MSHR表中有 block_addr 地址的条目。
*/
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  //在block_addr条目中，查找所有的mem_fetch行为。
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    //如果(*it)->is_write()为真，代表it是写行为，写请求正处于挂起状态。
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    //如果当前(*it)不是写行为，是读行为，但是write_found又为true，则之前有一个对 block_addr 地址
    //的写行为，因此存在对 block_addr 地址的写后读行为被挂起。
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/*
Accept a new cache fill response: mark entry ready for processing.
接受新的缓存填充响应：标记条目以备处理。这里假设的是MSHR表中有 block_addr 地址的条目。
*/
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  //busy（）始终返回false，此句无效。
  assert(!busy());
  //查找 block_addr 地址对应的条目。
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  //将对 block_addr 地址的访问合并到就绪内存访问列表中。m_current_response是就绪内存访问的列表。
  //m_current_response仅存储了就绪内存访问的地址。
  m_current_response.push_back(block_addr);
  //设置原子标志位。
  has_atomic = a->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

/*
Returns next ready access.
返回一个已经填入的就绪访问。
*/
mem_fetch *mshr_table::next_access() {
  //access_ready()的功能是，如果存在就绪访问，则返回true。这里是假定存在就绪内存访问。
  assert(access_ready());
  //返回就绪内存访问列表的首个条目的条目地址。m_current_response是就绪内存访问的列表。
  //m_current_response仅存储了就绪内存访问的地址。
  new_addr_type block_addr = m_current_response.front();
  assert(!m_data[block_addr].m_list.empty());
  //返回block_addr的合并的内存访问行为的首个请求，mem_fetch=m_list.front()。
  mem_fetch *result = m_data[block_addr].m_list.front();
  //将合并的内存访问行为的首个请求从列表里 pop 出去。
  m_data[block_addr].m_list.pop_front();
  if (m_data[block_addr].m_list.empty()) {
    //在将合并的内存访问行为的首个请求从列表里 pop 出去后，列表如果变空即该条目失效，需要擦除该条目。
    // release entry
    m_data.erase(block_addr);
    //下一个就绪访问得到后，就绪内存访问列表中把该次就绪访问的地址 pop 出去。m_current_response仅存
    //储了就绪内存访问的地址。
    m_current_response.pop_front();
  }
  return result;
}

void mshr_table::display(FILE *fp) const {
  fprintf(fp, "MSHR contents\n");
  for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
    unsigned block_addr = e->first;
    fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
            e->second.m_has_atomic, e->second.m_list.size());
    if (!e->second.m_list.empty()) {
      mem_fetch *mf = e->second.m_list.front();
      fprintf(fp, "%p :", mf);
      mf->print(fp);
    } else {
      fprintf(fp, " no memory requests???\n");
    }
  }
}
/***************************************************************** Caches
 * *****************************************************************/
cache_stats::cache_stats() {
  m_stats.resize(NUM_MEM_ACCESS_TYPE);
  m_stats_pw.resize(NUM_MEM_ACCESS_TYPE);
  m_fail_stats.resize(NUM_MEM_ACCESS_TYPE);
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_stats_pw[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_fail_stats[i].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear() {
  //
  // Zero out all current cache statistics
  //
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    std::fill(m_stats_pw[i].begin(), m_stats_pw[i].end(), 0);
    std::fill(m_fail_stats[i].begin(), m_fail_stats[i].end(), 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear_pw() {
  //
  // Zero out per-window cache statistics
  //
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    std::fill(m_stats_pw[i].begin(), m_stats_pw[i].end(), 0);
  }
}

void cache_stats::inc_stats(int access_type, int access_outcome) {
  //
  // Increment the stat corresponding to (access_type, access_outcome) by 1.
  //
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  m_stats[access_type][access_outcome]++;
}

void cache_stats::inc_stats_pw(int access_type, int access_outcome) {
  //
  // Increment the corresponding per-window cache stat
  //
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");
  m_stats_pw[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(int access_type, int fail_outcome) {
  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  m_fail_stats[access_type][fail_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  //
  // This function selects how the cache access outcome should be counted.
  // HIT_RESERVED is considered as a MISS in the cores, however, it should be
  // counted as a HIT_RESERVED in the caches.
  //
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome) {
  //
  // Simple method to read/modify the stat corresponding to (access_type,
  // access_outcome) Used overloaded () to avoid the need for separate
  // read/write member functions
  //
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
  }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome) const {
  //
  // Const accessor into m_stats.
  //
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
  }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  //
  // Overloaded + operator to allow for simple stat accumulation
  //
  cache_stats ret;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      ret(type, status, false) =
          m_stats[type][status] + cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      ret(type, status, true) =
          m_fail_stats[type][status] + cs(type, status, true);
    }
  }
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
  return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  //
  // Overloaded += operator to allow for simple stat accumulation
  //
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      m_stats[type][status] += cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      m_stats_pw[type][status] += cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      m_fail_stats[type][status] += cs(type, status, true);
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const {
  //
  // Print out each non-zero cache statistic for every memory access type and
  // status "cache_name" defaults to "Cache_stats" when no argument is
  // provided, otherwise the provided name is used. The printed format is
  // "<cache_name>[<request_type>][<request_status>] = <stat_value>"
  //
  std::vector<unsigned> total_access;
  total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
  std::string m_cache_name = cache_name;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_request_status_str((enum cache_request_status)status),
              m_stats[type][status]);

      if (status != RESERVATION_FAIL && status != MSHR_HIT)
        // MSHR_HIT is a special type of SECTOR_MISS
        // so its already included in the SECTOR_MISS
        total_access[type] += m_stats[type][status];
    }
  }
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    if (total_access[type] > 0)
      fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
              total_access[type]);
  }
}

void cache_stats::print_fail_stats(FILE *fout, const char *cache_name) const {
  std::string m_cache_name = cache_name;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS; ++fail) {
      if (m_fail_stats[type][fail] > 0) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type),
                cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                m_fail_stats[type][fail]);
      }
    }
  }
}

void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
  float data_port_util = 0.0f;
  if (port_available_cycles > 0) {
    data_port_util = (float)data_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
  float fill_port_util = 0.0f;
  if (port_available_cycles > 0) {
    fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  //
  // Returns a sum of the stats corresponding to each "access_type" and
  // "access_status" pair. "access_type" is an array of "num_access_type"
  // mem_access_types. "access_status" is an array of "num_access_status"
  // cache_request_statuses.
  //
  unsigned long long total = 0;
  for (unsigned type = 0; type < num_access_type; ++type) {
    for (unsigned status = 0; status < num_access_status; ++status) {
      if (!check_valid((int)access_type[type], (int)access_status[status]))
        assert(0 && "Unknown cache access type or access outcome");
      total += m_stats[access_type[type]][access_status[status]];
    }
  }
  return total;
}

void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
  //
  // Overwrites "css" with the appropriate statistics from this cache.
  //
  struct cache_sub_stats t_css;
  t_css.clear();

  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      if (status == HIT || status == MISS || status == SECTOR_MISS ||
          status == HIT_RESERVED)
        t_css.accesses += m_stats[type][status];

      if (status == MISS || status == SECTOR_MISS)
        t_css.misses += m_stats[type][status];

      if (status == HIT_RESERVED) t_css.pending_hits += m_stats[type][status];

      if (status == RESERVATION_FAIL) t_css.res_fails += m_stats[type][status];
    }
  }

  t_css.port_available_cycles = m_cache_port_available_cycles;
  t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
  t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

  css = t_css;
}

void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
  //
  // Overwrites "css" with the appropriate statistics from this cache.
  //
  struct cache_sub_stats_pw t_css;
  t_css.clear();

  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      if (status == HIT || status == MISS || status == SECTOR_MISS ||
          status == HIT_RESERVED)
        t_css.accesses += m_stats_pw[type][status];

      if (status == HIT) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_hits += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_hits += m_stats_pw[type][status];
        }
      }

      if (status == MISS || status == SECTOR_MISS) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_misses += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_misses += m_stats_pw[type][status];
        }
      }

      if (status == HIT_RESERVED) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_pending_hits += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_pending_hits += m_stats_pw[type][status];
        }
      }

      if (status == RESERVATION_FAIL) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_res_fails += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_res_fails += m_stats_pw[type][status];
        }
      }
    }
  }

  css = t_css;
}

bool cache_stats::check_valid(int type, int status) const {
  //
  // Verify a valid access_type/access_status
  //
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}

bool cache_stats::check_fail_valid(int type, int fail) const {
  //
  // Verify a valid access_type/access_status
  //
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
  m_cache_port_available_cycles += 1;
  if (data_port_busy) {
    m_cache_data_port_busy_cycles += 1;
  }
  if (fill_port_busy) {
    m_cache_fill_port_busy_cycles += 1;
  }
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
  m_data_port_occupied_cycles = 0;
  m_fill_port_occupied_cycles = 0;
}

// use the data port based on the outcome and events generated by the mem_fetch
// request
/*
根据mem_fetch请求生成的结果和事件使用数据端口。
*/
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events) {
  unsigned data_size = mf->get_data_size();
  unsigned port_width = m_config.m_data_port_width;
  switch (outcome) {
    case HIT: {
      unsigned data_cycles =
          data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
      m_data_port_occupied_cycles += data_cycles;
    } break;
    case HIT_RESERVED:
    case MISS: {
      // the data array is accessed to read out the entire line for write-back
      // in case of sector cache we need to write bank only the modified sectors
      cache_event ev(WRITE_BACK_REQUEST_SENT);
      if (was_writeback_sent(events, ev)) {
        unsigned data_cycles = ev.m_evicted_block.m_modified_size / port_width;
        m_data_port_occupied_cycles += data_cycles;
      }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      // Does not consume any port bandwidth
      break;
    default:
      assert(0);
      break;
  }
}

// use the fill port
/*
根据mem_fetch请求使用填充端口。
*/
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
  m_fill_port_occupied_cycles += fill_cycles;
}

// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles -= 1;
  }
  assert(m_data_port_occupied_cycles >= 0);

  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles -= 1;
  }
  assert(m_fill_port_occupied_cycles >= 0);
}

// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const {
  return (m_data_port_occupied_cycles == 0);
}

// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0);
}

// Sends next request to lower level of memory
/*
cache向前推进一拍。
*/
void baseline_cache::cycle() {
  //如果MISS请求队列中不为空，则将队首的请求发送到下一级内存。
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      /*************************************************************************************** tmp start */
      if (/* m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle && */
          get_sm_id() == PRINT_PROCESS_SM_ID && PRINT_EXECUTE_PROCESS) {
        printf("      * cache %s 's m_miss_queue is not empty, and m_memport is not full, "
              "just send m_miss_queue.front() to the next level of memory.\n", get_name().c_str());
      }
      /*************************************************************************************** tmp end   */
      m_miss_queue.pop_front();
      //mem_fetch_interface是对mem访存的接口。
      //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过
      //这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
      m_memport->push(mf);
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}

// Sends next request to lower level of memory
/*
cache向前推进一拍。
*/
void baseline_cache::cycle(unsigned long long cycle) {
  //如果MISS请求队列中不为空，则将队首的请求发送到下一级内存。
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      /*************************************************************************************** tmp start */
      if (/* m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle && */
          get_sm_id() == PRINT_PROCESS_SM_ID && PRINT_EXECUTE_PROCESS) {
        printf("      * cache %s 's m_miss_queue is not empty, and m_memport is not full, "
              "just send m_miss_queue.front() to the next level of memory.\n", get_name().c_str());
      }
      /*************************************************************************************** tmp end   */
      m_miss_queue.pop_front();
      //mem_fetch_interface是对mem访存的接口。
      //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过
      //这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
      m_memport->push(mf);
    }
    /*************************************************************************************** tmp start */
    else {
      if (PRINT_EXECUTE_STALL) {
        printf("Stall cycle[%llu]: Execute, SM-%d/wid-%d fails as m_memport of L1D is not free to put mf, "
               "insn pc[0x%04x]: ", 
               cycle, get_sm_id(), mf->get_inst().warp_id(), mf->get_inst().pc);
        mf->get_inst().print_sass_insn_line_tmp(stdout, mf->get_inst().warp_id(), mf->get_inst().pc);
        // fflush(stdout);
      }
    }
    /*************************************************************************************** tmp end   */
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}

// Interface for response from lower memory level (model bandwidth restictions
// in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  mf->set_data_size(e->second.m_data_size);
  mf->set_addr(e->second.m_addr);
  if (m_config.m_alloc_policy == ON_MISS)
    m_tag_array->fill(e->second.m_cache_index, time, mf);
  else if (m_config.m_alloc_policy == ON_FILL) {
    m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
	//if (m_config.is_streaming()) m_tag_array->remove_pending_line(mf);
  } else
    abort();
  bool has_atomic = false;
  m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
  if (has_atomic) {
    assert(m_config.m_alloc_policy == ON_MISS);
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as dirty for
                                                      // atomic operation
    block->set_byte_mask(mf);
  }
  m_extra_mf_fields.erase(mf);
  m_bandwidth_management.use_fill_port(mf);
}

// Checks if mf is waiting to be filled by lower memory level
/*
检查是否mf正在等待更低的存储层次填充。
*/
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  //extra_mf_fields_lookup的定义：
  //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
  //向cache发出数据请求mf时，如果未命中，且在MSHR中也未命中（没有mf条目），则将其加入到MSHR中，
  //同时，设置m_extra_mf_fields[mf]，意味着如果mf在m_extra_mf_fields中存在，即mf等待着DRAM
  //的数据回到L2缓存填充：
  //m_extra_mf_fields[mf] = extra_mf_fields(
  //      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
  fprintf(fp, "Cache %s:\t", m_name.c_str());
  m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const {
  fprintf(fp, "Cache %s:\n", m_name.c_str());
  m_mshrs.display(fp);
  fprintf(fp, "\n");
}

// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;
  evicted_block_info e;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e,
                    events, read_only, wa);
}

// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (mshr_hit && mshr_avail) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    m_mshrs.add(mshr_addr, mf);
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT);
    do_miss = true;

  } else if (!mshr_hit && mshr_avail &&
             (m_miss_queue.size() < m_config.m_miss_queue_size)) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    m_mshrs.add(mshr_addr, mf);
    //if (m_config.is_streaming() && m_config.m_cache_type == SECTOR) {
    //  m_tag_array->add_pending_line(mf);
    //}
    m_extra_mf_fields[mf] = extra_mf_fields(
        mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    mf->set_data_size(m_config.get_atom_sz());
    mf->set_addr(mshr_addr);
    //mf为miss的请求，加入miss_queue，MISS请求队列。
    //在baseline_cache::cycle()中，会将m_miss_queue队首的数据包mf传递给下一层缓存。
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
    if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));

    do_miss = true;
  } else if (mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
  else if (!mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
  else
    assert(0);
}

// Sends write request to lower level memory (write or writeback)
//将数据写请求一同发送至下一级存储。
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);
  //在baseline_cache::cycle()中，会将m_miss_queue队首的数据包mf传递给下一层缓存。
  m_miss_queue.push_back(mf);
  mf->set_status(m_miss_queue_status, time);
}

/*
更新一个cache block的状态为可读。如果所有的byte mask位全都设置为dirty了，则将该sector可
设置为可读，因为当前的sector已经是全部更新为最新值了，是可读的。这个函数对所有的数据请求mf
的所有访问的sector进行遍历，如果mf所访问的所有的byte mask位全都设置为dirty了，则将该cache
block设置为可读。
*/
void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  //这里传入的参数是cache block的index。
  // For example, 4 sets, 6 ways:
  // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
  // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
  // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
  // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
  //                |--------> index => cache_block_t *line
  cache_block_t *block = m_tag_array->get_block(cache_index);
  //对当前cache block的4个sector进行遍历。
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    //第i个sector被数据请求mf访问。
    if (mf->get_access_sector_mask().test(i)) {
      //all_set是指所有的byte mask位都被设置成了dirty了。
      bool all_set = true;
      //这里k是隶属于第i个sector的byte的编号。
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set, 
        // the sector is unreadble
        //如果第i个sector中有任意一个byte的dirty mask位没有被设置，则all_set就是false。
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      if (all_set)
        //如果所有的byte mask位全都设置为dirty了，则将该sector可设置为可读，因为当前的
        //sector已经是全部更新为最新值了，是可读的。
        block->set_m_readable(true, mf->get_access_sector_mask());
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

// Write-back hit: Mark block as modified
/*
若Write Hit时采取write-back策略，则需要将数据不单单写入cache，还需要直接将数据写入下一
级存储。
*/
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的
  //所有位。
  //这里write-back策略不需要直接将数据写入下一级存储，因此不需要调用miss_queue_full()
  //以及send_write_request()函数来发送请求到下一级存储。
  new_addr_type block_addr = m_config.block_addr(addr);
  //LRU状态的更新。
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  //更新一个cache block的状态为可读。如果所有的byte mask位全都设置为dirty了，则将该sector
  //可设置为可读，因为当前的sector已经是全部更新为最新值了，是可读的。这个函数对所有的数据请
  //求mf的所有访问的sector进行遍历，如果mf所访问的所有的byte mask位全都设置为dirty了，则将
  //该cache block设置为可读。
  update_m_readable(mf,cache_index);

  return HIT;
}

// Write-through hit: Directly send request to lower level memory
/*
若Write Hit时采取write-through策略，则需要将数据不单单写入cache，还需要直接将数据写入下
一级存储。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  //miss_queue_full检查是否一个miss请求能够在当前时钟周期内被处理，m_miss_queue_size在
  //V100的L1 cache中配置为16，在L2 cache中配置为32，当一个请求的大小大到m_miss_queue放
  //不下时，它就在当前时钟周期内无法处理，发生RESERVATION_FAIL。
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    //如果miss_queue满了，但由于write-through策略要求数据应该直接写入下一级存储，因此这
    //里返回RESERVATION_FAIL，表示当前时钟周期内无法处理该请求。
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  //LRU状态的更新。
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  //更新一个cache block的状态为可读。如果所有的byte mask位全都设置为dirty了，则将该sector
  //可设置为可读，因为当前的sector已经是全部更新为最新值了，是可读的。这个函数对所有的数据请
  //求mf的所有访问的sector进行遍历，如果mf所访问的所有的byte mask位全都设置为dirty了，则将
  //该cache block设置为可读。
  update_m_readable(mf,cache_index);

  // generate a write-through
  //write-through策略将数据写入下一级存储。
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return HIT;
}

// Write-evict hit: Send request to lower level memory and invalidate
// corresponding block
/*
V100中暂时没有配置该策略，暂时不管。
*/
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index);
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  // Invalidate block
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

// Global write-evict, local write-back: Useful for private caches
/*
V100中暂时没有配置该策略，暂时不管。
*/
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
}

/****** Write-miss functions (Set by config file) ******/

// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
/*
V100中暂时没有配置该策略，暂时不管。
*/
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (miss_queue_full(2) ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2))
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    else if (mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
    else if (!mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
    else
      assert(0);

    return RESERVATION_FAIL;
  }

  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

  mem_fetch *n_mf =
      new mem_fetch(*ma, NULL, mf->get_ctrl_size(), mf->get_wid(),
                    mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
                    m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      assert(status ==
             MISS);  // SECTOR_MISS and HIT_RESERVED should not send write back
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_parition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    return MISS;
  }

  return RESERVATION_FAIL;
}

/*
V100中暂时没有配置该策略，暂时不管。
*/
enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache block/sector, then, write and set
    // cache block Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    bool wb = false;
    evicted_block_info evicted;

    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    //在当前版本的GPGPU-Sim中，set_ignore_on_fill暂时用不到。
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_parition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  } else {
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(1))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
      else
        assert(0);

      return RESERVATION_FAIL;
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING);
      return RESERVATION_FAIL;
    }

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_ctrl_size(), mf->get_wid(), mf->get_sid(),
        mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);

    //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
    //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
    //有位。
    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_parition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }
}

/*
FETCH_ON_READ policy。 m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read。
需要参考 https://arxiv.org/pdf/1810.07269.pdf 论文对Volta架构访存行为的解释。
*/
enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache block/sector, then, write and set
  // cache block Modified. and no need to send read request to memory or reserve
  // mshr

  // FETCH_ON_READ policy: https://arxiv.org/pdf/1810.07269.pdf
  // In literature, there are two different write allocation policies [32], fetch-
  // on-write and write-validate. In fetch-on-write, when we write to a single byte
  // of a sector, the L2 fetches the whole sector then merges the written portion 
  // to the sector and sets the sector as modified. In the write-validate policy, 
  // no read fetch is required, instead each sector has a bit-wise write-mask. When 
  // a write to a single byte is received, it writes the byte to the sector, sets 
  // the corresponding write bit and sets the sector as valid and modified. When a 
  // modified cache block is evicted, the cache block is written back to the memory 
  // along with the write mask. It is important to note that, in a write-validate 
  // policy, it assumes the read and write granularity can be in terms of bytes in 
  // order to exploit the benefits of the write-mask. In fact, based on our micro-
  // benchmark shown in Figure 5, we have observed that the L2 cache applies some-
  // thing similar to write-validate. However, all the reads received by L2 caches 
  // from the coalescer are 32-byte sectored accesses. Thus, the read access granu-
  // larity (32 bytes) is different from the write access granularity (one byte). 
  // To handle this, the L2 cache applies a different write allocation policy, 
  // which we named lazy fetch-on-read, that is a compromise between write-validate 
  // and fetch-on-write. When a sector read request is received to a modified sector, 
  // it first checks if the sector write-mask is complete, i.e. all the bytes have 
  // been written to and the line is fully readable. If so, it reads the sector, 
  // otherwise, similar to fetch-on-write, it generates a read request for this 
  // sector and merges it with the modified bytes.

  //miss_queue_full检查是否一个miss请求能够在当前时钟周期内被处理，m_miss_queue_size在
  //V100的L1 cache中配置为16，在L2 cache中配置为32，当一个请求的大小大到m_miss_queue放
  //不下时，它就在当前时钟周期内无法处理完毕，发生RESERVATION_FAIL。
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  //在V100配置中，L1 cache为'T'-write through，L2 cache为'B'-write back。
  if (m_config.m_write_policy == WRITE_THROUGH) {
    //如果是write through，则需要将数据一同写回下一层存储。
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  }

  bool wb = false;
  evicted_block_info evicted;

  //更新LRU状态。Least Recently Used。
  //对一个cache进行数据访问的时候，调用data_cache::access()函数：
  //- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  //  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
  //- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  //  回的cache访问状态，执行相应的操作。
  //  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
  //    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
  //    函数来实现LRU状态的更新。
  //m_lines[idx]作为逐出并reserve新访问的cache block，如果它的某个sector已经被MODIFIED，则
  //需要执行写回操作，设置写回的标志为wb=true，并设置逐出cache block的信息。
  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

  // Theoretically, the passing parameter status should be the same as the m_status, 
  // if the assertion fails here, go to function `wr_miss_wa_lazy_fetch_on_read` to 
  // remove this assertion. yangjianchao16 add
  assert((m_status == status));
  assert(m_status != HIT);
  //cache_index是cache block的index。
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  //如果Cache block[mask]状态是RESERVED，说明有其他的线程正在读取这个Cache block。挂起的命
  //中访问已命中处于RESERVED状态的缓存行，这意味着同一行上已存在由先前缓存未命中发送的flying
  //内存请求。
  if (m_status == HIT_RESERVED) {
    //在当前版本的GPGPU-Sim中，set_ignore_on_fill暂时用不到。
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache 
    //block是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    //在FETCH_ON_READ policy: https://arxiv.org/pdf/1810.07269.pdf 中提到，访问cache发生
    //miss时：
    // In the write-validate policy, no read fetch is required, instead each sector has 
    // a bit-wise write-mask. When a write to a single byte is received, it writes the 
    // byte to the sector, sets the corresponding write bit and sets the sector as valid 
    // and modified. When a modified cache block is evicted, the cache block is written 
    // back to the memory along with the write mask.
    //而在FETCH_ON_READ中，需要设置sector的byte mask。这里就是指设置这个byte mask的标志。
    block->set_byte_mask_on_fill(true);
  }

  //m_config.get_atom_sz()为SECTOR_SIZE=4，即mf访问的是一整个sector=4字节。
  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    //由于mf访问的是整个sector，因此整个sector都是dirty的，设置访问的sector可读。
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    //由于mf访问的是部分sector，因此只有mf访问的那部分sector是dirty的，设置访问的sector不可读。
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
  }
  //更新一个cache block的状态为可读。如果所有的byte mask位全都设置为dirty了，则将该sector可
  //设置为可读，因为当前的sector已经是全部更新为最新值了，是可读的。这个函数对所有的数据请求mf
  //的所有访问的sector进行遍历，如果mf所访问的所有的byte mask位全都设置为dirty了，则将该cache
  //block设置为可读。
  update_m_readable(mf,cache_index);

  //m_status的状态可以为HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL。
  //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
  //被MODIFIED，来逐出；但是如果dirty的cache block的比例超过m_wr_percent（V100中配置为
  //25%），也可以不满足MODIFIED的条件。
  //all_reserved被初始化为true，是指所有cache block都没有能够逐出来为新访问提供RESERVE
  //的空间，这里一旦满足上面两个if条件，说明cache block可以被逐出来提供空间供RESERVE新访
  //问，这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明cache
  //line不可被逐出，发生RESERVATION_FAIL。这里不发生RESERVATION_FAIL，说明能够找到一个
  //cache block逐出并reserve新访问。
  if (m_status != RESERVATION_FAIL) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    //这里假设m_lines[idx]作为逐出并reserve新访问的cache block，如果它的某个sector已经
    //被MODIFIED，则需要执行写回操作，设置写回的标志为wb=true。
    //在V100配置中，L1 cache为'T'-write through，L2 cache为'B'-write back。这里L2
    //cache会进下面的if块。
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      //m_wrbk_type：L1 cache为L1_WRBK_ACC，L2 cache为L2_WRBK_ACC。
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_parition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

// No write-allocate miss: Simply send write request to lower level memory
/*
V100中暂时没有配置该策略，暂时不管。
*/
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  //miss_queue_full检查是否一个miss请求能够在当前时钟周期内被处理，m_miss_queue_size在
  //V100的L1 cache中配置为16，在L2 cache中配置为32，当一个请求的大小大到m_miss_queue放
  //不下时，它就在当前时钟周期内无法处理完毕，发生RESERVATION_FAIL。
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
/*
READ HIT操作。
*/
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  //原子操作从全局存储取值，计算，并写回相同地址三项事务在同一原子操作中完成，因此会修改
  //cache的状态为MODIFIED。
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    //获取该原子操作的cache block，并判断其是否先前已被MODIFIED，如果先前未被MODIFIED，此
    //次原子操作做出MODIFIED，要增加dirty数目，如果先前已经被MODIFIED，则先前dirty数目已
    //经增加过了，就不需要再增加了。
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    //设置cache block的状态为MODIFIED。
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as MODIFIED
    //设置dirty_byte_mask。
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
/*
READ MISS操作。
*/
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  //读miss时，就需要将数据请求发送至下一级存储。
  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }

  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_parition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access(
    new_addr_type addr, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write());
  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      if (do_miss)
        cache_status = MISS;
      else
        cache_status = RESERVATION_FAIL;
    } else {
      cache_status = RESERVATION_FAIL;
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status));
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status));
  return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
/*
一个通用函数，它获取tag_array探测的结果并根据缓存配置执行相应的功能。
access函数调用它：
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events);
  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
/*
L1 和 L2 目前使用相同的访问功能。两个缓存之间的区分是通过配置缓存策略来完成的。
L1 和 L2 都覆盖此函数，以提供在包含此类操作时执行特定于每个缓存的操作的方法。
对cache进行数据访问。

对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  //m_config.get_atom_sz()是cache替换原子操作的粒度，如果cache是SECTOR类型的，粒度为
  //SECTOR_SIZE，否则为line_size。
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  //cache_index会返回依据tag位选中的cache block的索引。
  unsigned cache_index = (unsigned)-1;
  //判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL等状态。
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status));
  m_stats.inc_stats_pw(mf->get_access_type(), m_stats.select_stats_status(
                                                  probe_status, access_status));
  return access_status;
}

// This is meant to model the first level data cache in Fermi.
// It is write-evict (global) or write-back (local) at the
// granularity of individual blocks (Set by GPGPU-Sim configuration file)
// (the policy used in fermi according to the CUDA manual)
/*
这是为了对Fermi中的第一级数据缓存进行建模。它是单个块粒度的写逐出（global）或写回（local）
（由 GPGPU-Sim 配置文件设置）（根据 CUDA 手册在Fermi中使用的策略）。
*/
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

// Access function for tex_cache
// return values: RESERVATION_FAIL if request could not be accepted
// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
// since unlike a normal CPU cache, a "HIT" in texture cache does not
// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    return RESERVATION_FAIL;

  assert(mf->get_data_size() <= m_config.get_line_sz());

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  //m_config.block_addr(addr): return addr & ~(new_addr_type)(m_line_sz - 1);
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tags.access(block_addr, time, cache_index, mf);
  enum cache_request_status cache_status = RESERVATION_FAIL;
  assert(status != RESERVATION_FAIL);
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  m_fragment_fifo.push(
      fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  if (status == MISS) {
    // we need to send a memory request...
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
    mf->set_data_size(m_config.get_line_sz());
    m_tags.fill(cache_index, time, mf);  // mark block as valid
    m_request_fifo.push(mf);
    mf->set_status(m_request_queue_status, time);
    events.push_back(cache_event(READ_REQUEST_SENT));
    cache_status = MISS;
  } else {
    // the value *will* *be* in the cache already
    cache_status = HIT_RESERVED;
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status));
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status));
  return cache_status;
}

void tex_cache::cycle() {
  // send next request to lower level of memory
  if (!m_request_fifo.empty()) {
    mem_fetch *mf = m_request_fifo.peek();
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      m_request_fifo.pop();
      //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是
      //通过这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
      m_memport->push(mf);
    }
  }
  // read ready lines from cache
  if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
    const fragment_entry &e = m_fragment_fifo.peek();
    if (e.m_miss) {
      // check head of reorder buffer to see if data is back from memory
      unsigned rob_index = m_rob.next_pop_index();
      const rob_entry &r = m_rob.peek(rob_index);
      assert(r.m_request == e.m_request);
      // assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr())
      // );
      if (r.m_ready) {
        assert(r.m_index == e.m_cache_index);
        m_cache[r.m_index].m_valid = true;
        m_cache[r.m_index].m_block_addr = r.m_block_addr;
        m_result_fifo.push(e.m_request);
        m_rob.pop();
        m_fragment_fifo.pop();
      }
    } else {
      // hit:
      assert(m_cache[e.m_cache_index].m_valid);
      assert(m_cache[e.m_cache_index].m_block_addr ==
             m_config.block_addr(e.m_request->get_addr()));
      m_result_fifo.push(e.m_request);
      m_fragment_fifo.pop();
    }
  }
}

// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  assert(!m_rob.empty());
  mf->set_status(m_rob_status, time);

  unsigned rob_index = e->second.m_rob_index;
  rob_entry &r = m_rob.peek(rob_index);
  assert(!r.m_ready);
  r.m_ready = true;
  r.m_time = time;
  assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const {
  fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
  fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
          m_fragment_fifo.capacity());
  fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
          m_rob.capacity());
  fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
          m_request_fifo.capacity());
  if (!m_rob.empty()) fprintf(fp, "reorder buffer contents:\n");
  for (int n = m_rob.size() - 1; n >= 0; n--) {
    unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
    const rob_entry &r = m_rob.peek(index);
    fprintf(fp, "tex rob[%3d] : %s ", index,
            (r.m_ready ? "ready  " : "pending"));
    if (r.m_ready)
      fprintf(fp, "@%6u", r.m_time);
    else
      fprintf(fp, "       ");
    fprintf(fp, "[idx=%4u]", r.m_index);
    r.m_request->print(fp, false);
  }
  if (!m_fragment_fifo.empty()) {
    fprintf(fp, "fragment fifo (oldest) :");
    fragment_entry &f = m_fragment_fifo.peek();
    fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
    f.m_request->print(fp, false);
  }
}
/******************************************************************************************************************************************/
