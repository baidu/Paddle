// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package paddle

// #include "pd_config.h"
// #include "pd_common.h"
// #include "pd_types.h"
// #include "pd_utils.h"
// #include <stdlib.h>
// #include <string.h>
import "C"
import (
	"unsafe"
)

type Precision C.PD_PrecisionType

const (
	PrecisionFloat32 Precision = C.PD_PRECISION_FLOAT32
	PrecisionInt8    Precision = C.PD_PRECISION_INT8
	PrecisionHalf    Precision = C.PD_PRECISION_HALF
)

type Config struct {
	c *C.PD_Config
}

///
/// \brief Create a new config.
///
func NewConfig() *Config {
	cConfig := C.PD_ConfigCreate()
	config := &Config{c: cConfig}
	return config
}

///
/// \brief Set the combined model with two specific paths for program and
/// parameters.
///
/// \param model model file path of the combined model.
/// \param params params file path of the combined model.
///
func (config *Config) SetModel(model, params string) {
	cmodel := C.CString(model)
	cparams := C.CString(params)
	C.PD_ConfigSetModel(config.c, cmodel, cparams)
	defer func() {
		C.free(unsafe.Pointer(cmodel))
		C.free(unsafe.Pointer(cparams))
	}()
}

///
/// \brief Set the no-combined model dir path.
///
/// \param modelDir model dir path.
///
func (config *Config) SetModelDir(modelDir string) {
	cmodel := C.CString(modelDir)
	C.PD_ConfigSetModelDir(config.c, cmodel)
	defer C.free(unsafe.Pointer(cmodel))
}

///
/// \brief Set the model file path of a combined model.
///
/// \param x model file path.
///
func (config *Config) SetProgFile(model string) {
	cmodel := C.CString(model)
	C.PD_ConfigSetProgFile(config.c, cmodel)
	defer C.free(unsafe.Pointer(cmodel))
}

///
/// \brief Set the params file path of a combined model.
///
/// \param x params file path.
///
func (config *Config) SetParamsFile(params string) {
	cparams := C.CString(params)
	C.PD_ConfigSetParamsFile(config.c, cparams)
	defer C.free(unsafe.Pointer(cparams))
}

///
/// \brief Set the path of optimization cache directory.
///
/// \param cacheDir the path of optimization cache directory.
///
func (config *Config) SetOptimCacheDir(cacheDir string) {
	ccacheDir := C.CString(cacheDir)
	C.PD_ConfigSetOptimCacheDir(config.c, ccacheDir)
	defer C.free(unsafe.Pointer(ccacheDir))
}

///
/// \brief Get the model directory path.
///
/// \return string The model directory path.
///
func (config *Config) ModelDir() string {
	return C.GoString(C.PD_ConfigGetModelDir(config.c))
}

///
/// \brief Get the program file path.
///
/// \return string The program file path.
///
func (config *Config) ProgFile() string {
	return C.GoString(C.PD_ConfigGetProgFile(config.c))
}

///
/// \brief Get the combined parameters file.
///
/// \return string The combined parameters file.
///
func (config *Config) ParamsFile() string {
	return C.GoString(C.PD_ConfigGetParamsFile(config.c))
}

///
/// \brief Turn off FC Padding.
///
func (config *Config) DisableFCPadding() {
	C.PD_ConfigDisableFCPadding(config.c)
}

///
/// \brief A boolean state telling whether fc padding is used.
///
/// \return bool Whether fc padding is used.
///
func (config *Config) UseFcPadding() bool {
	return cvtPDBoolToGo(C.PD_ConfigUseFcPadding(config.c))
}

///
/// \brief Turn on GPU.
///
/// \param memorySize initial size of the GPU memory pool in MB.
/// \param deviceId the GPU card to use.
///
func (config *Config) EnableUseGpu(memorySize uint64, deviceId int32) {
	C.PD_ConfigEnableUseGpu(config.c, C.uint64_t(memorySize), C.int32_t(deviceId), 0)
}

///
/// \brief Turn on ONNXRuntime.
///
func (config *Config) EnableONNXRuntime() {
	C.PD_ConfigEnableONNXRuntime(config.c)
}

///
/// \brief Turn off ONNXRuntime.
///
func (config *Config) DisableONNXRuntime() {
	C.PD_ConfigDisableONNXRuntime(config.c)
}

///
/// \brief A boolean state telling whether the ONNXRuntime is turned on.
///
/// \return bool Whether the ONNXRuntime is turned on.
///
func (config *Config) ONNXRuntimeEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigONNXRuntimeEnabled(config.c))
}

///
/// \brief Turn on ONNXRuntime Optimization.
///
func (config *Config) EnableORTOptimization() {
	C.PD_ConfigEnableORTOptimization(config.c)
}

///
/// \brief Turn on XPU.
///
/// \param l3Size The size of the video memory allocated by the l3 cache, the maximum is 16M.
/// \param l3Locked Whether the allocated L3 cache can be locked. If false, it means that the L3 cache is not locked, and the allocated L3 cache can be shared by multiple models, and multiple models sharing the L3 cache will be executed sequentially on the card.
/// \param convAutotune Whether to autotune the conv operator in the model. If true, when the conv operator of a certain dimension is executed for the first time, it will automatically search for a better algorithm to improve the performance of subsequent conv operators of the same dimension.
/// \param convAutotuneFile Specify the path of the autotune file. If autotune_file is specified, the algorithm specified in the file will be used and autotune will not be performed again.
/// \param transformerEencoderPrecision Calculation accuracy of multi_encoder
/// \param transformerEncoderAdaptiveSeqlen Is the input of multi_encoder variable length
/// \param enable_multi_stream Whether to enable the multi stream of xpu
///
func (config *Config) EnableXpu(l3Size int32, l3Locked bool, convAutotune bool, convAutotuneFile string, transformerEencoderPrecision string, transformerEncoderAdaptiveSeqlen bool, enableMultiStream bool) {
	cConvAutotuneFile := C.CString(convAutotuneFile)
	cTransformerEencoderPrecision := C.CString(transformerEencoderPrecision)
	defer func() {
		C.free(unsafe.Pointer(cConvAutotuneFile))
		C.free(unsafe.Pointer(cTransformerEencoderPrecision))
	}()
	C.PD_ConfigEnableXpu(config.c, C.int32_t(l3Size), cvtGoBoolToPD(l3Locked), cvtGoBoolToPD(convAutotune), cConvAutotuneFile, cTransformerEencoderPrecision, cvtGoBoolToPD(transformerEncoderAdaptiveSeqlen), cvtGoBoolToPD(enableMultiStream))
}

///
/// \brief A boolean state telling whether the GPU is turned on.
///
/// \return bool Whether the GPU is turned on.
///
func (config *Config) UseGpu() bool {
	return cvtPDBoolToGo(C.PD_ConfigUseGpu(config.c))
}

///
/// \brief A boolean state telling whether the XPU is turned on.
///
/// \return bool Whether the XPU is turned on.
///
func (config *Config) UseXpu() bool {
	return cvtPDBoolToGo(C.PD_ConfigUseXpu(config.c))
}

///
/// \brief Get the GPU device id.
///
/// \return int32 The GPU device id.
///
func (config *Config) GpuDeviceId() int32 {
	return int32(C.PD_ConfigGpuDeviceId(config.c))
}

///
/// \brief Get the XPU device id.
///
/// \return int32 The XPU device id.
///
func (config *Config) XpuDeviceId() int32 {
	return int32(C.PD_ConfigXpuDeviceId(config.c))
}

///
/// \brief Get the initial size in MB of the GPU memory pool.
///
/// \return int32 The initial size in MB of the GPU memory pool.
///
func (config *Config) MemoryPoolInitSizeMb() int32 {
	return int32(C.PD_ConfigMemoryPoolInitSizeMb(config.c))
}

///
/// \brief Get the proportion of the initial memory pool size compared to the
/// device.
///
/// \return float32 The proportion of the initial memory pool size.
///
func (config *Config) FractionOfGpuMemoryForPool() float32 {
	return float32(C.PD_ConfigFractionOfGpuMemoryForPool(config.c))
}

///
/// \brief Control whether to perform IR graph optimization.
/// If turned off, the AnalysisConfig will act just like a NativeConfig.
///
/// \param x Whether the ir graph optimization is active.
///
func (config *Config) SwitchIrOptim(x bool) {
	C.PD_ConfigSwitchIrOptim(config.c, cvtGoBoolToPD(x))
}

///
/// \brief A boolean state telling whether the ir graph optimization is
/// active.
///
/// \return bool Whether to use ir graph optimization.
///
// bool ir_optim() const { return enable_ir_optim_; }
func (config *Config) IrOptim() bool {
	return cvtPDBoolToGo(C.PD_ConfigIrOptim(config.c))
}

///
/// \brief Turn on the TensorRT engine.
/// The TensorRT engine will accelerate some subgraphes in the original Fluid
/// computation graph. In some models such as resnet50, GoogleNet and so on,
/// it gains significant performance acceleration.
///
/// \param workspaceSize The memory size(in byte) used for TensorRT
/// workspace.
/// \param maxBatchSize The maximum batch size of this prediction task,
/// better set as small as possible for less performance loss.
/// \param minSubgraphSize The minimum TensorRT subgraph size needed, if a
/// subgraph is smaller than this, it will not be transferred to TensorRT
/// engine.
/// \param precision The precision used in TensorRT.
/// \param useStatic Serialize optimization information to disk for reusing.
/// \param useCalibMode Use TRT int8 calibration(post training
/// quantization).
///
func (config *Config) EnableTensorRtEngine(workspaceSize int64, maxBatchSize int32, minSubgraphSize int32,
	precision Precision, useStatic bool, useCalibMode bool) {
	C.PD_ConfigEnableTensorRtEngine(config.c, C.int64_t(workspaceSize), C.int32_t(maxBatchSize), C.int32_t(minSubgraphSize), C.int32_t(precision), cvtGoBoolToPD(useStatic), cvtGoBoolToPD(useCalibMode))
}

///
/// \brief A boolean state telling whether the TensorRT engine is used.
///
/// \return bool Whether the TensorRT engine is used.
///
func (config *Config) TensorRtEngineEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigTensorRtEngineEnabled(config.c))
}

///
/// \brief Set min, max, opt shape for TensorRT Dynamic shape mode.
/// \param minInputShape The min input shape of the subgraph input.
/// \param maxInputShape The max input shape of the subgraph input.
/// \param optimInputShape The opt input shape of the subgraph input.
/// \param disableTrtPluginFp16 Setting this parameter to true means that
/// TRT plugin will not run fp16.
///
func (config *Config) SetTRTDynamicShapeInfo(minInputShape map[string][]int32, maxInputShape map[string][]int32,
	optimInputShape map[string][]int32, disableTrtPluginFp16 bool) {

	tensorNum := uint(len(minInputShape))
	names := make([](*C.char), tensorNum)
	goNames := make([]string, tensorNum)
	var shapeNum []uint

	idx := 0
	for n := range minInputShape {
		char := C.CString(n)
		defer C.free(unsafe.Pointer(char))
		names[idx] = (*C.char)(unsafe.Pointer(char))
		goNames[idx] = n
		shapeNum = append(shapeNum, uint(len(minInputShape[n])))
		idx++
	}

	cMinInputShape := make([]*C.int32_t, len(goNames))
	cMaxInputShape := make([]*C.int32_t, len(goNames))
	cOptInputShape := make([]*C.int32_t, len(goNames))
	for i, n := range goNames {
		pMin := (*C.int32_t)(C.malloc(C.size_t(C.sizeof_int32_t * len(minInputShape[n]))))
		cMinInputShape[i] = pMin

		// A []C.int32_t slice backed by C memory.
		// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
		// Using [1<<27] instead of [1<<30] so it works on 32-bit architecture
		pMinData := (*[1 << 27]C.int32_t)(unsafe.Pointer(pMin))
		for j, v := range minInputShape[n] {
			(*pMinData)[j] = C.int32_t(v)
		}
		defer C.free(unsafe.Pointer(pMin))

		pMax := (*C.int32_t)(C.malloc(C.size_t(C.sizeof_int32_t * len(maxInputShape[n]))))
		cMaxInputShape[i] = pMax
		pMaxData := (*[1 << 27]C.int32_t)(unsafe.Pointer(pMax))
		for j, v := range maxInputShape[n] {
			(*pMaxData)[j] = C.int32_t(v)
		}
		defer C.free(unsafe.Pointer(pMax))

		pOpt := (*C.int32_t)(C.malloc(C.size_t(C.sizeof_int32_t * len(optimInputShape[n]))))
		cOptInputShape[i] = pOpt
		pOptData := (*[1 << 27]C.int32_t)(unsafe.Pointer(pOpt))
		for j, v := range optimInputShape[n] {
			(*pOptData)[j] = C.int32_t(v)
		}
		defer C.free(unsafe.Pointer(pOpt))
	}

	C.PD_ConfigSetTrtDynamicShapeInfo(config.c, C.size_t(tensorNum), (**C.char)(unsafe.Pointer(&names[0])),
		(*C.size_t)(unsafe.Pointer(&shapeNum[0])),
		(**C.int32_t)(unsafe.Pointer(&cMinInputShape[0])),
		(**C.int32_t)(unsafe.Pointer(&cMaxInputShape[0])),
		(**C.int32_t)(unsafe.Pointer(&cOptInputShape[0])),
		cvtGoBoolToPD(disableTrtPluginFp16))
}

///
/// \brief A boolean state telling whether the trt dynamic_shape is used.
///
func (config *Config) TensorRtDynamicShapeEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigTensorRtDynamicShapeEnabled(config.c))
}

///
/// \brief Enable tuned tensorrt dynamic shape.
///
/// \param shapeRangeInfoPath the path to shape_info file got in
/// CollectShapeInfo mode.
/// \param allowBuildAtRuntime allow build trt engine at runtime.
///
func (config *Config) EnableTunedTensorRtDynamicShape(shapeRangeInfoPath string, allowBuildAtRuntime bool) {
	cstr := C.CString(shapeRangeInfoPath)
	C.PD_ConfigEnableTunedTensorRtDynamicShape(config.c, cstr, cvtGoBoolToPD(allowBuildAtRuntime))
	defer C.free(unsafe.Pointer(cstr))
}

///
/// \brief A boolean state telling whether to use tuned tensorrt dynamic
/// shape.
///
func (config *Config) TunedTensorRtDynamicShape() bool {
	return cvtPDBoolToGo(C.PD_ConfigTunedTensorRtDynamicShape(config.c))
}

///
/// \brief A boolean state telling whether to allow building trt engine at
/// runtime.
///
func (config *Config) TrtAllowBuildAtRuntime() bool {
	return cvtPDBoolToGo(C.PD_ConfigTrtAllowBuildAtRuntime(config.c))
}

///
/// \brief Collect shape info of all tensors in compute graph.
///
/// \param shapeRangeInfoPath the path to save shape info.
///
func (config *Config) CollectShapeRangeInfo(shapeRangeInfoPath string) {
	cstr := C.CString(shapeRangeInfoPath)
	C.PD_ConfigCollectShapeRangeInfo(config.c, cstr)
	defer C.free(unsafe.Pointer(cstr))
}

///
/// \brief the shape info path in CollectShapeInfo mode.
/// Attention, Please release the string manually.
///
func (config *Config) ShapeRangeInfoPath() string {
	cstr := C.PD_ConfigShapeRangeInfoPath(config.c)
	str := C.GoString(cstr)
	C.free(unsafe.Pointer(cstr))
	return str
}

///
/// \brief A boolean state telling whether to collect shape info.
///
func (config *Config) ShapeRangeInfoCollected() bool {
	return cvtPDBoolToGo(C.PD_ConfigShapeRangeInfoCollected(config.c))
}

///
/// \brief Prevent ops running in Paddle-TRT
/// NOTE: just experimental, not an official stable API, easy to be broken.
///
func (config *Config) DisableTensorRtOPs(ops []string) {
	num := uint(len(ops))
	var buf = make([]*C.char, num+1)
	for i, _ := range ops {
		char := C.CString(ops[i])
		defer C.free(unsafe.Pointer(char))
		buf[i] = (*C.char)(unsafe.Pointer(char))
	}

	C.PD_ConfigDisableTensorRtOPs(config.c, C.size_t(num), (**C.char)(unsafe.Pointer(&buf[0])))
}

///
/// \brief Replace some TensorRT plugins to TensorRT OSS(
/// https://github.com/NVIDIA/TensorRT), with which some models's inference
/// may be more high-performance. Libnvinfer_plugin.so greater than
/// V7.2.1 is needed.
///
func (config *Config) EnableVarseqlen() {
	C.PD_ConfigEnableVarseqlen(config.c)
}

///
/// \brief A boolean state telling whether to use the TensorRT OSS.
///
/// \return bool Whether to use the TensorRT OSS.
///
func (config *Config) TensorrtOssEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigTensorRtOssEnabled(config.c))
}

///
/// \brief Enable TensorRT DLA
/// \param dlaCore ID of DLACore, which should be 0, 1,
///        ..., IBuilder.getNbDLACores() - 1
///
func (config *Config) EnableTensorRtDLA(dlaCore int32) {
	C.PD_ConfigEnableTensorRtDla(config.c, C.int32_t(dlaCore))
}

///
/// \brief A boolean state telling whether to use the TensorRT DLA.
///
/// \return bool Whether to use the TensorRT DLA.
///
func (config *Config) TensorrtDlaEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigTensorRtDlaEnabled(config.c))
}

///
/// \brief Control whether to debug IR graph analysis phase.
/// This will generate DOT files for visualizing the computation graph after
/// each analysis pass applied.
///
/// \param x whether to debug IR graph analysis phase.
///
func (config *Config) SwitchIrDebug(x bool) {
	C.PD_ConfigSwitchIrDebug(config.c, cvtGoBoolToPD(x))
}

///
/// \brief Turn on OneDNN.
///
func (config *Config) EnableMKLDNN() {
	C.PD_ConfigEnableMKLDNN(config.c)
}

///
/// \brief Set the cache capacity of different input shapes for OneDNN.
/// Default value 0 means not caching any shape.
/// Please see MKL-DNN Data Caching Design Document:
/// https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md
///
/// \param capacity The cache capacity.
///
func (config *Config) SetMkldnnCacheCapacity(capacity int32) {
	C.PD_ConfigSetMkldnnCacheCapacity(config.c, C.int32_t(capacity))
}

///
/// \brief A boolean state telling whether to use the OneDNN.
///
/// \return bool Whether to use the OneDNN.
///
func (config *Config) MkldnnEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigMkldnnEnabled(config.c))
}

///
/// \brief Set the number of cpu math library threads.
///
/// \param mathThreadsNum The number of cpu math library
/// threads.
///
func (config *Config) SetCpuMathLibraryNumThreads(mathThreadsNum int) {
	C.PD_ConfigSetCpuMathLibraryNumThreads(config.c, C.int32_t(mathThreadsNum))
}

///
/// \brief An int state telling how many threads are used in the CPU math
/// library.
///
/// \return int The number of threads used in the CPU math library.
///
func (config *Config) CpuMathLibraryNumThreads() int32 {
	return int32(C.PD_ConfigGetCpuMathLibraryNumThreads(config.c))
}

///
/// \brief Transform the AnalysisConfig to NativeConfig.
///
/// \return NativeConfig The NativeConfig transformed.
///
// NativeConfig ToNativeConfig() const;

///
/// \brief Specify the operator type list to use OneDNN acceleration.
///
/// \param opList The operator type list.
///
func (config *Config) SetMKLDNNOp(opList []string) {
	num := uint(len(opList))
	// Add one in case num is zero.
	var buf = make([]*C.char, num+1)
	for i, _ := range opList {
		char := C.CString(opList[i])
		defer C.free(unsafe.Pointer(char))
		buf[i] = (*C.char)(unsafe.Pointer(char))
	}

	C.PD_ConfigSetMkldnnOp(config.c, C.size_t(num), (**C.char)(unsafe.Pointer(&buf[0])))
}

///
/// \brief Turn on OneDNN bfloat16.
///
func (config *Config) EnableMkldnnBfloat16() {
	C.PD_ConfigEnableMkldnnBfloat16(config.c)
}

///
/// \brief A boolean state telling whether to use the OneDNN Bfloat16.
///
/// \return bool Whether to use the OneDNN Bfloat16.
///
func (config *Config) MkldnnBfloat16Enabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigMkldnnBfloat16Enabled(config.c))
}

/// \brief Specify the operator type list to use Bfloat16 acceleration.
///
/// \param opList The operator type list.
///
func (config *Config) SetBfloat16Op(opList []string) {
	num := uint(len(opList))
	// Add one in case num is zero.
	var buf = make([]*C.char, num+1)
	for i, _ := range opList {
		char := C.CString(opList[i])
		defer C.free(unsafe.Pointer(char))
		buf[i] = (*C.char)(unsafe.Pointer(char))
	}

	C.PD_ConfigSetBfloat16Op(config.c, C.size_t(num), (**C.char)(unsafe.Pointer(&buf[0])))
}

///
/// \brief A boolean state telling whether the thread local CUDA stream is
/// enabled.
///
/// \return bool Whether the thread local CUDA stream is enabled.
///
func (config *Config) ThreadLocalStreamEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigThreadLocalStreamEnabled(config.c))
}

///
/// \brief Specify the memory buffer of program and parameter.
/// Used when model and params are loaded directly from memory.
///
/// \param prog The memory buffer of program.
/// \param params The memory buffer of the combined parameters file.
///
func (config *Config) SetModelBuffer(prog, params string) {
	cProg := C.CString(prog)
	cParams := C.CString(params)
	defer func() {
		C.free(unsafe.Pointer(cProg))
		C.free(unsafe.Pointer(cParams))
	}()

	C.PD_ConfigSetModelBuffer(config.c, cProg, C.size_t(len(prog)), cParams, C.size_t(len(params)))
}

///
/// \brief A boolean state telling whether the model is set from the CPU
/// memory.
///
/// \return bool Whether model and params are loaded directly from memory.
///
func (config *Config) ModelFromMemory() bool {
	return cvtPDBoolToGo(C.PD_ConfigModelFromMemory(config.c))
}

///
/// \brief Turn on memory optimize
/// NOTE still in development.
///
func (config *Config) EnableMemoryOptim(x bool) {
	C.PD_ConfigEnableMemoryOptim(config.c, cvtGoBoolToPD(x))
}

///
/// \brief A boolean state telling whether the memory optimization is
/// activated.
///
/// \return bool Whether the memory optimization is activated.
///
func (config *Config) MemoryOptimEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigMemoryOptimEnabled(config.c))
}

///
/// \brief Turn on profiling report.
/// If not turned on, no profiling report will be generated.
///
func (config *Config) EnableProfile() {
	C.PD_ConfigEnableProfile(config.c)
}

///
/// \brief A boolean state telling whether the profiler is activated.
///
/// \return bool Whether the profiler is activated.
///
func (config *Config) ProfileEnabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigProfileEnabled(config.c))
}

///
/// \brief Mute all logs in Paddle inference.
///
func (config *Config) DisableGlogInfo() {
	C.PD_ConfigDisableGlogInfo(config.c)
}

///
/// \brief A boolean state telling whether logs in Paddle inference are muted.
///
/// \return bool Whether logs in Paddle inference are muted.
///
func (config *Config) GlogInfoDisabled() bool {
	return cvtPDBoolToGo(C.PD_ConfigGlogInfoDisabled(config.c))
}

///
/// \brief A boolean state telling whether the AnalysisConfig is valid.
///
/// \return bool Whether the AnalysisConfig is valid.
///
func (config *Config) IsValid() bool {
	return cvtPDBoolToGo(C.PD_ConfigIsValid(config.c))
}

///
/// \brief Enable the GPU multi-computing stream feature.
/// NOTE: The current behavior of this interface is to bind the computation
/// stream to the thread, and this behavior may be changed in the future.
///
func (config *Config) EnableGpuMultiStream() {
	C.PD_ConfigEnableGpuMultiStream(config.c)
}

///
/// \brief Delete all passes that has a certain type 'pass'.
///
/// \param[in] pass the certain pass type to be deleted.
///
func (config *Config) DeletePass(pass string) {
	cPass := C.CString(pass)
	C.PD_ConfigDeletePass(config.c, cPass)
	C.free(unsafe.Pointer(cPass))
}

///
/// \brief Append a pass to the end of the passes
///
/// \param[in] pass the new pass.
///
func (config *Config) AppendPass(pass string) {
	cPass := C.CString(pass)
	C.PD_ConfigAppendPass(config.c, cPass)
	C.free(unsafe.Pointer(cPass))
}

///
/// \brief  Insert a pass to a specific position
///
/// \param[in] idx the position to insert.
/// \param[in] pass the new pass.
///
func (config *Config) InsertPass(idx uint64, pass string) {
	cPass := C.CString(pass)
	C.PD_ConfigInsertPass(config.c, C.size_t(idx), cPass)
	C.free(unsafe.Pointer(cPass))
}

///
/// \brief Get information of passes.
///
/// \return Return list of the passes.
///
func (config *Config) AllPasses() []string {
	cPasses := C.PD_ConfigAllPasses(config.c)
	num := int(cPasses.size)
	passes := cvtToGoSliceString(num, cPasses.data)
	C.PD_OneDimArrayCstrDestroy(cPasses)
	return passes
}

///
/// \brief Get information of config.
///
/// \return Return config info.
///
func (config *Config) Summary() string {
	cSummary := C.PD_ConfigSummary(config.c)
	summary := C.GoString(cSummary.data)
	C.PD_CstrDestroy(cSummary)
	return summary
}
