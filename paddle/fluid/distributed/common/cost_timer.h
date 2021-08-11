#pragma once
#include <memory>
#include <unordered_map>
#include "glog/logging.h"
#include "butil/time.h"
#include "bvar/latency_recorder.h"

namespace paddle {
namespace distributed {
    struct CostProfilerNode {
        std::shared_ptr<bvar::LatencyRecorder> recorder;
    };
    
    class CostProfiler {
    public:
        ~CostProfiler() {}
        static CostProfiler& instance() {
            static CostProfiler profiler;
            return profiler;
        }

        void register_profiler(const std::string& label) {
            if (_cost_profiler_map.find(label) != _cost_profiler_map.end()) {
                return;
            }
            auto profiler_node = std::make_shared<CostProfilerNode>();
            profiler_node->recorder.reset(new bvar::LatencyRecorder("cost_profiler", label));
            _cost_profiler_map[label] = profiler_node; 
        }
<<<<<<< HEAD

=======
        
>>>>>>> xf_test_psc
        CostProfilerNode* profiler(const std::string& label) {
            auto itr = _cost_profiler_map.find(label);
            if (itr != _cost_profiler_map.end()) {
                return itr->second.get();
            }
            return NULL;
        }
<<<<<<< HEAD

=======
        
>>>>>>> xf_test_psc
    private:
        CostProfiler() {}
        std::unordered_map<std::string, std::shared_ptr<CostProfilerNode>> _cost_profiler_map;
    };
<<<<<<< HEAD

=======
    
>>>>>>> xf_test_psc
    class CostTimer {
    public:
        CostTimer(const std::string& label) {
            _label = label;
            auto& profiler = CostProfiler::instance();
            _profiler_node = profiler.profiler(label);
            //如果不在profiler中，则使用log输出耗时信息
            _is_print_cost = _profiler_node == NULL;
            _start_time_ms = butil::gettimeofday_ms();
<<<<<<< HEAD

=======
            
>>>>>>> xf_test_psc
        }
        CostTimer(CostProfilerNode& profiler_node) {
            _is_print_cost = false;
            _profiler_node = &profiler_node;
            _start_time_ms = butil::gettimeofday_ms();
        }
        ~CostTimer() {
            if (_is_print_cost) {
                LOG(INFO) << "CostTimer label:" << _label 
                    << ", cost:" << butil::gettimeofday_ms() - _start_time_ms << "ms";
            } else {
                *(_profiler_node->recorder) << butil::gettimeofday_ms() - _start_time_ms;
            }
        }

    private:
        std::string _label;
        bool _is_print_cost;
        uint64_t _start_time_ms;
        CostProfilerNode* _profiler_node;
    };
}
<<<<<<< HEAD
} 
=======
}
>>>>>>> xf_test_psc
