// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/image/io.hpp>
#include <aliceVision/sfm/utils/alignment.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/config.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <sstream>
#include <vector>


// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 4
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string referenceFolder;
    std::string inputFolder;
    std::string outputFolder;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()("input,i", po::value<std::string>(&inputFolder)->required(), "Input folder to compare.")(
        "output,o", po::value<std::string>(&outputFolder)->required(), "Output folder to save differences.")(
        "reference,r", po::value<std::string>(&referenceFolder)->required(), "Reference folder to compare with.");

    CmdLine cmdline("AliceVision compareFolders");
    cmdline.add(requiredParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    fs::path inputPath(inputFolder);
    fs::path referencePath(referenceFolder);
    fs::path outputPath(outputFolder);
    if (!fs::exists(outputPath))
    {
        fs::create_directories(outputPath);
    }

    auto toLower = [](std::string string) -> std::string
    {
        auto copy = string;
        std::transform(copy.begin(), copy.end(), copy.begin(), [](unsigned char c) { return std::tolower(c); });
        return copy;
    };

    fs::recursive_directory_iterator end;
    for (fs::recursive_directory_iterator i(inputPath); i != end; ++i)
    {
        const auto& path = i->path();
        auto ext = toLower(path.extension().string());
        if(ext == ".exr")
        {
            std::cout << "compared filed: " << path.filename().string() << std::endl;
            auto reference = referencePath / path.filename();

            if (!fs::exists(reference))
            {
                std::cout << "\tequivalent file not found" << std::endl;
            }
            else
            {
                image::Image<float> image, imageRef;
                image::readImage(path.string(), image, image::EImageColorSpace::LINEAR);
                image::readImage(reference.string(), imageRef, image::EImageColorSpace::LINEAR);

                if (image.Width() != imageRef.Width() || image.Height() != imageRef.Height())
                {
                    std::cout << "\tequivalent file does not have the same size" << std::endl;
                }
                else
                {
                    image::RowMatrixXf errors = (image - imageRef).cwiseAbs();
                    double mean = errors.mean();
                    double min = errors.minCoeff();
                    double max = errors.maxCoeff();
                    std::cout << "\tmean error: " << mean << std::endl;
                    std::cout << "\tmin error: " << min << std::endl;
                    std::cout << "\tmax error: " << max << std::endl;

                    if (max != 0)
                    {
                        errors *= 255.0 / max;
                    }

                    auto outputfile = outputPath / path.filename().replace_extension(".jpg");
                    image::Image<unsigned char> output = {errors.cast<unsigned char>()};
                    oiio::ParamValueList metadata;
                    image::writeImage(outputfile.string(), output,
                                      image::ImageWriteOptions().toColorSpace(image::EImageColorSpace::NO_CONVERSION),
                                      metadata);
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

