#pragma once
#include <string>
#include <cassert>
#include <cstdint>
#include <istream>
#include <vector>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py=pybind11;


inline float cast_to_float(Half v)
{
    return DirectX::PackedVector::XMConvertHalfToFloat(v);
}

inline float cast_to_float(float v)
{
    return v;
}

enum class DescType
{
    eSrv,
    eUav
};

class NodeDispatcher
{
public:
    virtual std::uint32_t get_total_descriptor_count() = 0;

    virtual void initialize(ID3D12GraphicsCommandList* cmd_list, 
                            D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, 
                            D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) = 0;

    virtual void execute(ID3D12GraphicsCommandList* cmd_list) = 0;

    virtual void compile_shader(const std::string &cm_file, const std::string &build_options)=0;

    virtual std::vector<MType> get_output_vector(ID3D12CommandQueue* command_queue,
                                                ID3D12CommandAllocator* command_allocator,
                                                ID3D12GraphicsCommandList* command_list) = 0;

    virtual ~NodeDispatcher() = default;
};


inline ComPtr<ID3D12RootSignature> create_root_signature(ID3D12Device* d3d12_device, std::vector<DescType> desc_list)
{
    const auto bindings_size = desc_list.size();
    std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
    std::vector<CD3DX12_ROOT_PARAMETER1> root_params;
    ranges.reserve(bindings_size);
    root_params.reserve(bindings_size + 1); // + 1 beacuse of the CM driver path

    std::uint32_t srv_range_reg = 0;
    std::uint32_t uav_range_reg = 0;
    std::uint32_t cbv_range_reg = 0;

    {
        // driver thing
        CD3DX12_ROOT_PARAMETER1 rp{};
        rp.InitAsConstants(1, cbv_range_reg++);
        root_params.push_back(rp);
    }

    auto add_desc_table = [&](DescType type)
    {
        if (type == DescType::eSrv)
        {
            ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, srv_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE });
        }
        else
        {
            ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, uav_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE });
        }
        CD3DX12_ROOT_PARAMETER1 rp{};
        rp.InitAsDescriptorTable(1u, &ranges.back());
        root_params.push_back(rp);
    };

    for (const auto d : desc_list)
    {
        add_desc_table(d);
    }

    if (root_params.size() == 0)
    {
        throw std::runtime_error("Something gone wrong. Why kernel has 0 root params?");
    }

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_signature_desc;
    compute_root_signature_desc.Init_1_1(static_cast<UINT>(root_params.size()), root_params.data(), 0, nullptr);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    throw_if_failed(D3DX12SerializeVersionedRootSignature(
        &compute_root_signature_desc,
        D3D_ROOT_SIGNATURE_VERSION_1_1,
        &signature,
        &error), "D3DX12SerializeVersionedRootSignature failed.");

    if (error)
    {
        throw_with_msg("Failed to create root signature, error:" + std::string((LPCSTR)error->GetBufferPointer()));
    }
    ComPtr<ID3D12RootSignature> ret;
    throw_if_failed(d3d12_device->CreateRootSignature(
        0,
        signature->GetBufferPointer(),
        signature->GetBufferSize(),
        IID_PPV_ARGS(&ret)), "CreateRootSignature(...) failed.");
    return ret;
}


inline std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> create_resource_views_and_handles(ID3D12Device* d3d12_device, std::vector<std::pair<DescType, ID3D12Resource*>> resources_list, 
                                                                                    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
{
    const auto desc_heap_incrs_size = d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    const auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
    const auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles;
    gpu_handles.reserve(resources_list.size());

    for (std::size_t i = 0; i < resources_list.size(); i++)
    {
        auto cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(base_cpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size);
        gpu_handles.push_back(CD3DX12_GPU_DESCRIPTOR_HANDLE(base_gpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size));

        auto& resource_view_type = resources_list[i].first;
        auto& resource = resources_list[i].second;
        assert(resource != nullptr);
        const auto res_desc = resource->GetDesc();
        assert(res_desc.Dimension == D3D12_RESOURCE_DIMENSION::D3D12_RESOURCE_DIMENSION_BUFFER);

        if (resource_view_type == DescType::eSrv)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8_UINT;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
            desc.Buffer.FirstElement = 0;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

            d3d12_device->CreateShaderResourceView(resource, &desc, cpu_handle);
        }
        else
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
            desc.Buffer.FirstElement = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

            d3d12_device->CreateUnorderedAccessView(resource, nullptr, &desc, cpu_handle);
        }
    }

    return gpu_handles;
}


inline void dispatch_kernel(ID3D12GraphicsCommandList* cmd_list, ID3D12PipelineState* pso, ID3D12RootSignature* root_signature, 
                            std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles, std::uint32_t thg_x, std::uint32_t thg_y, std::uint32_t thg_z)
{
    assert(thg_x > 0);
    assert(thg_y > 0);
    assert(thg_z > 0);
    assert(cmd_list);
    assert(root_signature);
    assert(pso);
    assert(!gpu_handles.empty());

    cmd_list->SetComputeRootSignature(root_signature);
    cmd_list->SetPipelineState(pso);

    uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
    for (uint32_t i = 0; i < gpu_handles.size(); i++)
    {
        const auto gpu_heap_handle = gpu_handles[i];
        cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
    }

    cmd_list->Dispatch(thg_x, thg_y, thg_z);
}