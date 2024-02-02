

#include <chrono>
#include <format>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>

#include "rt_igdext.h"
#include "dx12_utils.h"
#include "cm_dispatcher.h"

inline void print_performance_stats(const std::vector<std::chrono::microseconds>& timings)
{
    std::chrono::microseconds avg(0);
    std::chrono::microseconds best((std::numeric_limits<uint32_t>::max)());
    std::chrono::microseconds median(0);

    // avg and best
    {
        for (const auto& t : timings)
        {
            avg += t;
            if (t < best)
            {
                best = t;
            }
        }
        avg /= timings.size();
    }

    // median
    {
      auto timings_copy = timings;
      std::nth_element(timings_copy.begin(), timings_copy.begin() + timings_copy.size() / 2, timings_copy.end());
      median = timings_copy[timings_copy.size() / 2];
    }

    std::cout << "Avg: " << avg.count() << std::endl;
    std::cout << "Median: " << avg.count() << std::endl;
    std::cout << "Best: " << best.count() << std::endl;
}


std::vector<MType> test_rt_igdext(const std::string &cm_file, const std::string &build_options,
                                const py::args &args, const py::kwargs &kwargs) {

    assert(kwargs.contains("iter_nums"));

    std::cout<< "---------cm_file: " << cm_file << std::endl;
    std::cout<< "---build_options: " << build_options << std::endl;
    std::cout<< "---iter_nums: " << kwargs["iter_nums"].cast<std::uint32_t>() << std::endl;
    std::cout<< "------------------ form buffers ------------------" << std::endl;
    // constexpr const std::uint32_t MAX_ITERATIONS = kwargs["iter_nums"].cast<u_int>();
    // constexpr const std::uint32_t dispatch_iterations = kwargs["iter_nums"].cast<u_int>();
    std::uint32_t MAX_ITERATIONS = kwargs["iter_nums"].cast<std::uint32_t>();
    std::uint32_t dispatch_iterations = kwargs["iter_nums"].cast<std::uint32_t>();
    std::vector<MType> result;


    try
    {
        // generic type of layers options
        // specific for implementation
        ComPtr<ID3D12Device> d3d12_device;
        ComPtr<ID3D12CommandQueue> command_queue;
        ComPtr<ID3D12CommandAllocator> command_allocator;
        ComPtr<ID3D12GraphicsCommandList> command_list;
        initalize_d3d12(d3d12_device, command_queue, command_allocator, command_list);
        auto dml_device = create_dml_device(d3d12_device.Get());
        auto performance_collector = initialize_d3d12_performance_collector(d3d12_device.Get(), MAX_ITERATIONS);

        auto intel_extension_d3d12 = IntelExtension(d3d12_device.Get());

        // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
        ComPtr<IDMLCommandRecorder> dml_command_recorder;
        throw_if_failed(dml_device->CreateCommandRecorder(IID_PPV_ARGS(dml_command_recorder.ReleaseAndGetAddressOf())), "create dml command recorder");


        std::unique_ptr<NodeDispatcher> node;
        node = std::make_unique<CmDispatcher>(intel_extension_d3d12, d3d12_device.Get(), dml_device.Get(), 
                                            dml_command_recorder.Get(), command_list.Get(), args, kwargs);

        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());
        node->compile_shader(cm_file, build_options);
        
        // bind descriptor heap
        const auto descriptors_count = node->get_total_descriptor_count();
        auto descriptor_heap = create_descriptor_heap(d3d12_device.Get(), descriptors_count);
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        // initalize
        node->initialize(command_list.Get(), descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        // Bind
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        // Execute & measure the operator on the GPU.
        for (std::uint32_t i = 0; i < dispatch_iterations; ++i)
        {
            performance_collector.add_timestamp(command_list.Get());
            node->execute(command_list.Get());
            performance_collector.add_timestamp(command_list.Get());
        }
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        
        const auto device_remove_reason = d3d12_device->GetDeviceRemovedReason();
        if (device_remove_reason != S_OK) {
          printf("Device removal. Reason: %d\n", device_remove_reason);
        }

        result = node->get_output_vector(command_queue.Get(), command_allocator.Get(), command_list.Get());

        // Copy the timing data back
        command_list->ResolveQueryData(
            performance_collector.timestamp_query_heap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            0,
            performance_collector.timestamp_index,
            performance_collector.timestamp_readback_buffer.Get(),
            0);
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        // Perfromance statistic
        uint64_t timestamp_frequency = 0;
        command_queue->GetTimestampFrequency(&timestamp_frequency);
        const auto timestamps_timings = get_timestamps_timings_from_ptr<std::chrono::microseconds>(timestamp_frequency, performance_collector.timestamp_readback, performance_collector.timestamp_index);
        performance_collector.timestamp_index = 0;
        std::vector<std::chrono::microseconds> timings(timestamps_timings.size() / 2);
        for (uint32_t i = 0; i < timings.size(); i++) {
          const auto t0 = timestamps_timings[i * 2];
          const auto t1 = timestamps_timings[i * 2 + 1];
          timings[i] = t1 - t0;
        }
        print_performance_stats(timings);

    } catch (std::exception e){
        std::cerr << "Exception caught: {" << e.what() << "} \n";
        return result;
    }

    return result;
}

