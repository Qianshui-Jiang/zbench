#pragma once

#include <iostream>
#include <optional>
#include <format>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py=pybind11;

#include "dx12_utils.h"
#include "layers_utils.h"


class CmDispatcher : public NodeDispatcher
{
public:
    CmDispatcher(IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, 
                IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list,
                const py::args &args, const py::kwargs &kwargs)
    : intc_ext_(intc_ext),d3d12_device_(d3d12_device), dml_device_(dml_device),dml_cmd_recorder_(dml_cmd_recorder)
    {   
        std::size_t total_upload_size = 0;
        assert(kwargs.contains("thg_x"));
        assert(kwargs.contains("thg_y"));
        assert(kwargs.contains("thg_z"));
        if (kwargs) {
            thg_x_ = kwargs["thg_x"].cast<u_int>();
            thg_y_ = kwargs["thg_y"].cast<u_int>();
            thg_z_ = kwargs["thg_z"].cast<u_int>();
            // get total upload size 
            for (auto item : kwargs) {
                // std::cout << "buffer key: " << item.first
                //             << " , type: " << py::type::of(item.second).str()
                //             << std::endl; // <class 'numpy.ndarray'>
                if (py::type::of(item.second) == py::type::of(py::array())) {
                    auto buf_in = py::array_t<MType, py::array::c_style | py::array::forcecast>(kwargs[item.first]);
                    all_tensor_arrays_.push_back(buf_in);
                    py::buffer_info buf_info = buf_in.request();
                    total_upload_size += buf_info.size * sizeof(MType);
                }
            }
        }

        // create upload buffer        
        all_io_buffers_.resize(all_tensor_arrays_.size());
        upload_buffer_ = create_buffer(d3d12_device, total_upload_size,
                                      D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_STATE_GENERIC_READ);

        // create data buffer

        for (int i = 0; i < all_io_buffers_.size(); i++) {
            if (py::type::of(all_tensor_arrays_[i]) == py::type::of(py::array())) {
                py::buffer_info tensor_info = all_tensor_arrays_[i].request();
                auto buffer_size_in_bytes = tensor_info.size * sizeof(MType);
                all_io_buffers_[i] = create_buffer(
                    d3d12_device, buffer_size_in_bytes, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                printf("tensor input [%d] > 0\n", i);
            }
        }

        // copy data into upload buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        for (int i=0; i< all_io_buffers_.size()-1; i++) {
            if (py::type::of(all_tensor_arrays_[i]) == py::type::of(py::array())) {
                // std::cout << all_tensor_arrays_.size() << std::endl;
                py::buffer_info tensor_info = all_tensor_arrays_[i].request();
                auto buffer_size_in_bytes = tensor_info.size * sizeof(MType);
                std::memcpy(upload_mapped_ptr + memcopy_offset, all_tensor_arrays_[i].data(), buffer_size_in_bytes);
                memcopy_offset += buffer_size_in_bytes;
            }
        }
    
        auto *ptr = reinterpret_cast<const Half*>(upload_mapped_ptr);
        // for(int i=0; i<4096*4096 + 4096; i++){
        //     assert(cast_to_float(ptr[i])==1);
        //     // printf("upload_mapped_ptr res: %f\n", cast_to_float(ptr[i]));
        // }
        upload_buffer_->Unmap(0, nullptr);

        // add cmd: copy upload buffer to input DX12 buffer 
        memcopy_offset = 0;
        for(int i=0; i< all_io_buffers_.size()-1; i++){
            // std::cout << "all_io_buffers:" << all_io_buffers_.size() << std::endl;
            py::buffer_info buf_info = all_tensor_arrays_[i].request();
            auto buffer_size_in_bytes = buf_info.size * sizeof(MType);
            cmd_list->CopyBufferRegion(all_io_buffers_[i].Get(), 0, upload_buffer_.Get(), memcopy_offset, buffer_size_in_bytes);
            memcopy_offset += buffer_size_in_bytes;
        }

        // cmd barries for copy data
        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        for(int i=0; i< all_io_buffers_.size()-1; i++){
            // std::cout << all_io_buffers_.size() << std::endl;
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(all_io_buffers_[i].Get(),
                            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());

        // got root_signature_
        std::vector<DescType> desc_list = {
            DescType::eSrv, // input a
            DescType::eSrv, // input b
            DescType::eUav  // output
        };
        root_signature_ = create_root_signature(d3d12_device, desc_list);

    }

    // void compile_shader(const char* cm_file, const char* build_options){
    void compile_shader(const std::string &cm_file, const std::string &build_options){
        assert(build_options != "None");
        if (build_options != "None") {
            std::cout << "build options: " << build_options << std::endl;
        }

        auto kernel_source_content = [&]()
        {
            std::string path = cm_file;
            std::fstream file(path);
            if (!file.is_open())
            {
                std::ostringstream oss;
                oss << "Kernel file can't be opened:" << path << " \n.";
                const auto msg = oss.str();
                throw std::runtime_error(msg);
            }
            return std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
        }();

        CD3DX12_SHADER_BYTECODE byte_code;
        byte_code.pShaderBytecode = kernel_source_content.data();
        byte_code.BytecodeLength = kernel_source_content.size();
        pso_ = intc_ext_.create_pipeline(byte_code, build_options, root_signature_.Get(), INTC_D3D12_SHADER_INPUT_TYPE::CM);
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        for(int i=0; i< all_io_buffers_.size()-1; i++){
            resources_list.push_back({ DescType::eSrv, all_io_buffers_[i].Get() });
        }
        resources_list.push_back({ DescType::eUav, all_io_buffers_.back().Get() });
        gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, cpu_handle, gpu_handle);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        cmd_list->SetComputeRootSignature(root_signature_.Get());
        cmd_list->SetPipelineState(pso_.Get());

        uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
        for (uint32_t i = 0; i < gpu_handles_.size(); i++)
        {
            const auto gpu_heap_handle = gpu_handles_[i];
            cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
        }

        cmd_list->Dispatch(thg_x_, thg_y_, thg_z_);
    }


    std::vector<MType> get_output_vector(ID3D12CommandQueue* command_queue, ID3D12CommandAllocator* command_allocator, 
                                            ID3D12GraphicsCommandList* command_list) override {
        const auto tensor_out_bytes_width = get_output_buffer_size();

        // readback data and validate
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(
            all_io_buffers_.back().Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), all_io_buffers_.back().Get());
        // std::cout<< "------------------ testing A ------------------" << std::endl;
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);
        // std::cout<< "------------------ testing B------------------" << std::endl;
        std::vector<std::byte> data_out(tensor_out_bytes_width);
        std::byte* readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
        readback_buffer->Unmap(0, nullptr);

        std::vector<MType> result(reinterpret_cast<MType*>(data_out.data()), reinterpret_cast<MType*>(data_out.data() + data_out.size()));
        const auto* gpu_typed_result = reinterpret_cast<const Half*>(data_out.data());

        // compare results
        printf("gpu res: %f\n", cast_to_float(gpu_typed_result[0]));
        // for (std::uint32_t i = 0; i < data_out.size() / sizeof(Half); i+=2)
        // {
        //     printf("gpu res[%d]: %f\n", i, cast_to_float(gpu_typed_result[i]));
        // }

        return result;
    }

protected:
    std::uint32_t get_total_descriptor_count() override
    {
        // input_a, input_b, output
        return all_io_buffers_.size();
    }

    size_t get_output_buffer_size() const
    {
        py::buffer_info buf_info = all_tensor_arrays_.back().request();
        size_t output_size = buf_info.size * sizeof(MType);
        return output_size;
    }

private:
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    ComPtr<ID3D12Resource> upload_buffer_;
    std::vector<ComPtr<ID3D12Resource>> all_io_buffers_;
    std::vector<py::array_t<MType, py::array::c_style | py::array::forcecast>> all_tensor_arrays_;

    // ---------------------------------
    u_int thg_x_;
    u_int thg_y_;
    u_int thg_z_;
    // ---------------------------------
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;
};
