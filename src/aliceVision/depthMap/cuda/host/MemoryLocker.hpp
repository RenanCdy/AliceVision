
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
        m_range = std::make_shared<sycl::range<Dim>>(m_hostMemory.getSize().y(), m_hostMemory.getSize().x());
        assert(m_hostMemory.getPitch() % sizeof(Type) == 0);
        sycl::range<Dim> bufferRange(m_hostMemory.getSize().y(), m_hostMemory.getPitch() / sizeof(Type));
        m_buffer = std::make_shared<sycl::buffer<SyclType, Dim>>(
            reinterpret_cast<SyclType*>(m_hostMemory.getBytePtr()), bufferRange);
        
    }

    ~BufferLocker()
    {
        m_deviceMemory.copyFrom(m_hostMemory, 0);
    }

    sycl::range<Dim>& range() { return *m_range; }
    sycl::buffer<SyclType, Dim>& buffer() { return *m_buffer; }

private:
    CudaDeviceMemoryPitched<Type, Dim>& m_deviceMemory;
    CudaHostMemoryHeap<Type, Dim> m_hostMemory;
    std::shared_ptr<sycl::range<Dim>> m_range;
    std::shared_ptr<sycl::buffer<SyclType, Dim>> m_buffer;
};

template <typename Type = CudaRGBA, unsigned Dim = 2>
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

    ~ImageLocker()
    {
        m_deviceMemory.copyFrom(m_hostMemory, 0);
    }

    sycl::range<Dim>& range() { return *m_range; }
    sycl::image<Dim>& image() { return *m_image; }

private:
    CudaDeviceMemoryPitched<Type, Dim>& m_deviceMemory;
    CudaHostMemoryHeap<Type, Dim> m_hostMemory;
    std::shared_ptr<sycl::range<Dim>> m_range;
    std::shared_ptr<sycl::image<Dim>> m_image;
};

} // namespace depthMap
} // namespace aliceVision
