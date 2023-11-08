
#include <aliceVision/depthMap/cuda/host/memory.hpp>
#include <sycl/sycl.hpp>

namespace aliceVision
{
namespace depthMap
{

template <typename T>
struct sycl_type_mapper {using type = T;};

template <> struct sycl_type_mapper<CudaRGBA>   { using type = sycl::ushort4; };

template <typename Type, unsigned Dim=2, typename SyclType = typename sycl_type_mapper<Type>::type>
class BufferLocker
{
public:
    BufferLocker(CudaDeviceMemoryPitched<Type, Dim>& deviceMemory)
        : m_deviceMemory(deviceMemory)
    {
        assert(sizeof(Type) == sizeof(SyclType));

        m_hostMemory.allocate(m_deviceMemory.getSize());
        m_hostMemory.copyFrom(m_deviceMemory, 0);
        m_range = std::make_shared<sycl::range<Dim>>( make_range<Dim>(m_hostMemory) );
        assert(m_hostMemory.getPitch() % sizeof(Type) == 0);
        sycl::range<Dim> bufferRange = make_buffer_range<Dim>(m_hostMemory);
        m_buffer = std::make_shared<sycl::buffer<SyclType, Dim>>(
            reinterpret_cast<SyclType*>(m_hostMemory.getBytePtr()), bufferRange);
        
    }

    BufferLocker(const CudaDeviceMemoryPitched<Type, Dim>& deviceMemory)
        : BufferLocker( const_cast<CudaDeviceMemoryPitched<Type, Dim>&>(deviceMemory) )
    {     
        m_cudaBufferIsConst = true;
    }


    ~BufferLocker()
    {
        if (!m_cudaBufferIsConst)
        {
            m_deviceMemory.copyFrom(m_hostMemory, 0);
        }
    }


    sycl::range<Dim>& range() { return *m_range; }
    sycl::buffer<SyclType, Dim>& buffer() { return *m_buffer; }

private:
    template <int D>
    sycl::range<D> make_range(const CudaHostMemoryHeap<Type, D>& hostMemory) {}

    template <>
    sycl::range<2> make_range<2>(const CudaHostMemoryHeap<Type, 2>& hostMemory) {
        return sycl::range<2>(hostMemory.getSize().y(), hostMemory.getSize().x());
    }

    template <>
    sycl::range<3> make_range<3>(const CudaHostMemoryHeap<Type, 3>& hostMemory) {
        return sycl::range<3>(hostMemory.getSize().z(), hostMemory.getSize().y(), hostMemory.getSize().x());
    }

    template <int D>
    sycl::range<D> make_buffer_range(const CudaHostMemoryHeap<Type, D>& hostMemory) {}

    template <>
    sycl::range<2> make_buffer_range<2>(const CudaHostMemoryHeap<Type, 2>& hostMemory) {
        return sycl::range<2>(hostMemory.getSize().y(), hostMemory.getPitch() / sizeof(Type));
    }

    template <>
    sycl::range<3> make_buffer_range<3>(const CudaHostMemoryHeap<Type, 3>& hostMemory) {
        return sycl::range<3>(hostMemory.getSize().z(), hostMemory.getSize().y(), hostMemory.getPitch() / sizeof(Type));
    }

    CudaDeviceMemoryPitched<Type, Dim>& m_deviceMemory;
    CudaHostMemoryHeap<Type, Dim> m_hostMemory;
    std::shared_ptr<sycl::range<Dim>> m_range;
    std::shared_ptr<sycl::buffer<SyclType, Dim>> m_buffer;
    bool m_cudaBufferIsConst = false;
};


template <typename Type = CudaRGBA, unsigned Dim = 2, typename SyclType = typename sycl_type_mapper<Type>::type>
class ImageLocker
{
public:
    ImageLocker(CudaDeviceMemoryPitched<Type, Dim>& deviceMemory)
        : m_deviceMemory(deviceMemory)
    {
        m_hostMemory.allocate(m_deviceMemory.getSize());
        m_hostMemory.copyFrom(m_deviceMemory, 0);
        m_range = std::make_shared<sycl::range<Dim>>(m_hostMemory.getSize().x(), m_hostMemory.getSize().y());
        m_image = std::make_shared<sycl::image<Dim>>(m_hostMemory.getBytePtr(), sycl::image_channel_order::rgba,
                                                   sycl::image_channel_type::fp16, *m_range, m_hostMemory.getPitch());

    }

    ImageLocker(const CudaDeviceMemoryPitched<Type, Dim>& deviceMemory)
    : ImageLocker( const_cast<CudaDeviceMemoryPitched<Type, Dim>&>(deviceMemory) )
    {
        m_cudaBufferIsConst = true;
    }

    ~ImageLocker()
    {
        if (!m_cudaBufferIsConst)
        {
            m_deviceMemory.copyFrom(m_hostMemory, 0);
        }
    }

    sycl::range<Dim>& range() { return *m_range; }
    sycl::image<Dim>& image() { return *m_image; }

private:
    CudaDeviceMemoryPitched<Type, Dim>& m_deviceMemory;
    CudaHostMemoryHeap<Type, Dim> m_hostMemory;
    std::shared_ptr<sycl::range<Dim>> m_range;
    std::shared_ptr<sycl::image<Dim>> m_image;
    bool m_cudaBufferIsConst = false;
};

} // namespace depthMap
} // namespace aliceVision
