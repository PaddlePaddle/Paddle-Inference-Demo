package main

import (
	"flag"
	"log"
	"sort"
	"sync"
	"time"

	pd "github.com/jiweibo/paddle/paddle/fluid/inference/goapi"
)

// flags
var modelName = flag.String("model_name", "resnet50/inference.pdmodel", "the model path")
var paramsName = flag.String("params_name", "resnet50/inference.pdiparams", "the params path")
var threadNum = flag.Int("thread_num", 4, "thread_num")
var workNum = flag.Int("work_num", 20, "work_num")
var batchSize = flag.Int("batch_size", 1, "batch size")

var cpuMath = flag.Int("cpu_math", 1, "cpu_math")
var useGpu = flag.Bool("use_gpu", false, "use_gpu")
var gpuId = flag.Int("gpu_id", 0, "gpu_id")
var useTrt = flag.Bool("use_trt", false, "use_trt")
var useTrtDynamicShape = flag.Bool("use_trt_dynamic_shape", false, "use_trt_dynamic_shape")

func main() {
	flag.Parse()

	var ch = make(chan int, *threadNum)
	for i := 0; i < *threadNum; i++ {
		ch <- i
	}
	var wg sync.WaitGroup
	var mx sync.Mutex
	var times []time.Duration

	config := pd.NewConfig()
	config.SetModel(*modelName, *paramsName)
	if *useGpu {
		config.EnableUseGpu(100, int32(*gpuId))
		if *useTrt {
			config.EnableTensorRtEngine(1<<30, 16, 3, pd.PrecisionFloat32, false, false)
			if *useTrtDynamicShape {
				minInputShape := make(map[string][]int32)
				minInputShape["inputs"] = []int32{int32(*batchSize), 3, 100, 100}
				maxInputShape := make(map[string][]int32)
				maxInputShape["inputs"] = []int32{int32(*batchSize), 3, 608, 608}
				optInputShape := make(map[string][]int32)
				optInputShape["inputs"] = []int32{int32(*batchSize), 3, 224, 224}
				config.SetTRTDynamicShapeInfo(minInputShape, maxInputShape, optInputShape, false)
			}
		}
	} else {
		config.SetCpuMathLibraryNumThreads(*cpuMath)
	}
	mainPredictor := pd.NewPredictor(config)
	predictors := []*pd.Predictor{}
	predictors = append(predictors, mainPredictor)
	for i := 0; i < *threadNum-1; i++ {
		predictors = append(predictors, mainPredictor.Clone())
	}

	for i := 0; i < *workNum; i++ {
		keyId := <-ch
		wg.Add(1)
		go func(predictors []*pd.Predictor, keyId int) {
			start := time.Now()

			inNames := predictors[keyId].GetInputNames()
			inHandle := predictors[keyId].GetInputHandle(inNames[0])
			outNames := predictors[keyId].GetOutputNames()
			outHandle := predictors[keyId].GetOutputHandle(outNames[0])

			data := make([]float32, 1*3*224*224)
			for i := 0; i < len(data); i++ {
				data[i] = float32(i%255) * 0.1
			}
			inHandle.Reshape([]int32{1, 3, 224, 224})
			inHandle.CopyFromCpu(data)

			predictors[keyId].Run()

			outData := make([]float32, numElements(outHandle.Shape()))
			outHandle.CopyToCpu(outData)
			tim := time.Now().Sub(start)

			log.Println("out max val:", maxValue(outData))
			mx.Lock()
			times = append(times, tim)

			defer func() {
				wg.Done()
				mx.Unlock()
				ch <- keyId
			}()
		}(predictors, keyId)
	}

	wg.Wait()
	timeInfo(times)
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}

func maxValue(vals []float32) (max float32) {
	max = 0
	for _, v := range vals {
		if v > max {
			max = v
		}
	}
	return
}

func timeInfo(times []time.Duration) {
	if len(times) == 1 {
		log.Printf("Only 1 time:%+v\n", times[0])
		return
	}
	sort.Slice(times, func(i, j int) bool {
		return times[i] < times[j]
	})
	req_percent := []float32{0.9, 0.95, 0.99}
	for _, p := range req_percent {
		idx := int32(float32(len(times))*p) - 1
		log.Printf("percent %v, cost time %v\n", p, times[idx])
	}
	var avg time.Duration = 0
	for _, t := range times {
		avg += t
	}
	log.Printf("avg time %vms\n", float32(avg.Nanoseconds()/1e6)/float32(len(times)))
}
