import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

// 常量
const INPUT_WIDTH = 224; // 输入图片宽高
const INPUT_HEIGHT = 224;
const EPOCHS = 10;
const CATEGORY_NAMES = ["墙", "脸", "手"]; // 分类类别名称

// 变量
let categoryIndex = -1; // 采集类别的index
let trainingDataInputs = []; // 训练集的经过左侧模型后得到的特征数据
let trainingDataOutputs = []; // 训练集的标签，即对应类别的index
let sampleCountList = CATEGORY_NAMES.map(() => 0); // 记录每个类别有多少个样本，如[98, 102, 83]
let modelLeft = null; // 定义左侧模型
let modelRight = null; // 定义右侧模型，只含FC层
let timerId = null;

function App() {
  const [pageStatus, setPageStatus] = useState("模型加载中，请稍后");
  const [step, setStep] = useState(0); // 1. 开启摄像头 2. 采集数据 3. 训练中、预测中

  // video元素
  const videoRef = useRef(null);

  // 点击开启摄像头
  function enableCam() {
    // 检查浏览器是否支持getUserMedia
    if (Boolean(navigator.mediaDevices?.getUserMedia)) {
      // 摄像头采集参数
      const constraints = {
        video: true,
        width: 640,
        height: 480,
      };

      // 启动摄像头，并在video元素中播放
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        if (videoRef.current) {
          // react video jsx不支持srcObject，只能通过ref传入
          videoRef.current.srcObject = stream;
        }
      });
    } else {
      console.warn("您的浏览器不支持 getUserMedia()");
    }
  }

  // 数据采集
  function dataGatherLoop() {
    const imageFeatures = tf.tidy(function () {
      // 将视频帧转换成tensor
      const videoFrameAsTensor = tf.browser.fromPixels(videoRef.current);
      // resize到224 * 244
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [INPUT_HEIGHT, INPUT_WIDTH],
        true
      );
      // 归一化到[0, 1]区间
      const normalizedTensorFrame = resizedTensorFrame.div(255);
      // 返回特征向量：增加1个batch维度，喂给前半个模型，得到特征向量，移除batch维度，[1024]
      return modelLeft.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    // 将数据和标签保存起来
    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(categoryIndex);

    // 类别样本数+1
    sampleCountList[categoryIndex]++;

    // 展示每个类别的样本数
    setPageStatus(
      CATEGORY_NAMES.map(
        (name, index) => `${name} 样本数：${sampleCountList[index] || 0}。`
      ).join()
    );
  }

  // 按下采集数据按钮，注意按下时不要移动鼠标
  function startCollectData(index) {
    // 更新数据采集状态：按下按钮时设置为类别id，松开时设置为-1
    categoryIndex = index;
    timerId = setInterval(dataGatherLoop, 20);
  }

  // 松开采集数据按钮
  function stopCollectData() {
    clearInterval(timerId);
  }

  // 预测画面中的物体
  function predictLoop() {
    tf.tidy(function () {
      // 对video的图像做归一化和resize
      const videoFrameAsTensor = tf.browser
        .fromPixels(videoRef.current)
        .div(255);
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [INPUT_HEIGHT, INPUT_WIDTH],
        true
      );
      // 预测图像类别
      const imageFeatures = modelLeft.predict(resizedTensorFrame.expandDims()); // [1, 1024]
      const prediction = modelRight.predict(imageFeatures).squeeze(); // [3]
      const highestIndex = prediction.argMax().arraySync();
      const predictionArray = prediction.arraySync();
      // 展示预测结果
      setPageStatus(
        `预测结果：${CATEGORY_NAMES[highestIndex]}，${Math.floor(
          predictionArray[highestIndex] * 100
        )}% 可信度`
      );
    });
  }

  // 点击开始训练
  async function trainAndPredict() {
    setPageStatus("模型训练中，进度0%");
    setStep(3);
    // 避免阻塞UI
    setTimeout(async () => {
      // 打乱训练集
      tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
      // 将训练集标签转换为tensor
      const outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
      // 训练集标签oneHot编码，[N, 3]
      const oneHotOutputs = tf.oneHot(outputsAsTensor, CATEGORY_NAMES.length);
      // 将输入数据从tensor1d数组堆叠成tensor2d，[N, 1024]
      const inputsAsTensor = tf.stack(trainingDataInputs);
      // 训练FC层
      await modelRight.fit(inputsAsTensor, oneHotOutputs, {
        shuffle: true,
        batchSize: 5,
        epochs: EPOCHS,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            // 打印进度
            setPageStatus(`模型训练中，进度${((epoch + 1) / EPOCHS) * 100}%`);
            console.log("Data for epoch " + epoch, logs);
          },
        },
      });
      // 清理内存
      outputsAsTensor.dispose();
      oneHotOutputs.dispose();
      inputsAsTensor.dispose();
      // 开始预测
      timerId = setInterval(predictLoop, 20);
    }, 0);
  }

  // 点击重置
  // function reset() {
  //     window.clearInterval(timerId)
  //     sampleCountList = []
  //     trainingDataInputs.forEach(x => x.dispose()) // 清理内存
  //     trainingDataInputs = []
  //     trainingDataOutputs = []
  //
  //     setPageStatus('请采集数据')
  //
  //     console.log('Tensors in memory: ' + tf.memory().numTensors)
  // }

  // 加载左侧模型，不含FC层
  useEffect(() => {
    (async () => {
      try {
        modelLeft = await tf.loadGraphModel("indexeddb://my-model-1");
      } catch {
        const URL =
          "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
        modelLeft = await tf.loadGraphModel(URL, { fromTFHub: true });
        await modelLeft.save("indexeddb://my-model-1");
      }

      setStep(1);
      setPageStatus("模型加载完成，请点击开启摄像头");

      // 预热模型：传零tensor，预先编译运算模块，加载权重到GPU，减少后面使用时的等待时间
      tf.tidy(function () {
        const answer = modelLeft.predict(
          tf.zeros([1, INPUT_HEIGHT, INPUT_WIDTH, 3])
        );
        console.log(answer.shape); // [1, 1024]，batchSize: 1，1024个特征
      });
    })();
  }, []);

  // 构建右侧模型
  useEffect(() => {
    modelRight = tf.sequential();
    modelRight.add(
      tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
    ); // 输入层
    modelRight.add(
      tf.layers.dense({ units: CATEGORY_NAMES.length, activation: "softmax" })
    ); // 分类需要使用softmax激活函数

    modelRight.summary(); // 打印模型构成

    modelRight.compile({
      optimizer: "adam", // adam会随时间改变学习率
      loss:
        CATEGORY_NAMES.length === 2
          ? "binaryCrossentropy"
          : "categoricalCrossentropy", // 区分两类还是多个类别
      metrics: ["accuracy"], // 记录准确率
    });
  }, []);

  return (
    <div className="App">
      <h1>基于MobileNet v3的图像分类</h1>
      {/* 应用状态 */}
      <p>{pageStatus}</p>
      {/* 展示摄像头采集的视频 */}
      {step >= 1 && (
        <video
          ref={videoRef}
          autoPlay
          muted
          onLoadedData={() => {
            setStep(2);
            setPageStatus("请采集图像"); // 变更页面状态
          }}
        />
      )}
      {/* 开启摄像头按钮 */}
      {step === 1 && <button onClick={enableCam}>开启摄像头</button>}
      {/* 采集数据按钮 */}
      {step === 2 &&
        CATEGORY_NAMES.map((name, index) => (
          <button
            key={name}
            onMouseDown={() => {
              startCollectData(index);
            }}
            onMouseUp={stopCollectData}
          >
            采集 {name} 的图像
          </button>
        ))}
      {/* 开始训练按钮 */}
      {step === 2 && <button onClick={trainAndPredict}>训练 + 预测</button>}
      {/* 重置按钮 */}
      {/*<button onClick={reset}>重置</button>*/}
    </div>
  );
}

export default App;
