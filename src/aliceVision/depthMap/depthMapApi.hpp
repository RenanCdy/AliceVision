#pragma once

#if defined(_Windows) || defined(_WINDOWS) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
	#ifdef aliceVision_depthMap_EXPORTS
	#define aliceVision_depthMap_DLL_API __declspec(dllexport)
	#else
	#define aliceVision_depthMap_DLL_API __declspec(dllimport)
	#endif
#else
	#define aliceVision_depthMap_DLL_API
#endif