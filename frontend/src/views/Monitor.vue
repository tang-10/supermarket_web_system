<template>
  <div class="h-[calc(100vh-64px)] p-4 flex flex-col gap-4 max-w-[1600px] mx-auto">
    <!-- 顶部：模式与源控制栏 -->
    <div class="bg-white p-4 rounded-xl shadow-sm border flex items-center gap-4">
      <div class="flex items-center gap-2">
        <span class="font-bold text-gray-700">模式:</span>
        <select
          v-model="mode"
          :disabled="isRunning"
          class="border p-2 rounded-lg outline-none cursor-pointer"
        >
          <option value="camera">实时摄像头</option>
          <option value="file">测试视频文件</option>
          <option value="ipcam">手机 IP 摄像头</option>
        </select>
      </div>

      <div v-if="mode === 'file'" class="flex items-center gap-2">
        <input
          ref="fileInputRef"
          type="file"
          @change="handleFileUpload"
          class="text-sm border p-1 rounded"
          accept="video/*"
        />
        <span class="text-xs text-gray-500">{{ uploadStatus || videoFileName || '未选择文件' }}</span>

        <!-- 【新增】输入手机流地址的输入框 -->
        <div v-if="mode === 'ipcam'" class="flex items-center gap-2">
          <input v-model="ipCamUrl" type="text" 
           :disabled="isRunning"
           class="bg-zinc-800 border border-zinc-700 text-zinc-100 p-2 rounded-lg outline-none w-64 text-sm" 
           placeholder="http://10.144.144.4:8080/video" />
        </div>
      </div>

      <div class="flex items-center gap-2">
        <button
          @click="toggleRecognition"
          class="px-6 py-2 rounded-lg font-bold transition-colors"
          :class="
            isRunning
              ? isPaused
                ? 'bg-green-500 text-white'
                : 'bg-yellow-500 text-white'
              : 'bg-blue-600 text-white'
          "
        >
          {{ !isRunning ? '开始识别' : isPaused ? '恢复侦测' : '暂停侦测' }}
        </button>
        <button
          v-if="isRunning || videoFileName"
          @click="clearVideoSource"
          class="px-4 py-2 bg-red-500 text-white rounded-lg font-bold hover:bg-red-600 transition-colors"
        >
          结束侦测并清空
        </button>
      </div>
    </div>

    <!-- 中部：视频渲染区 -->
    <div class="flex-1 flex gap-4 overflow-hidden">
      <!-- 视频流容器 -->
      <div class="flex-[2] relative bg-black rounded-xl overflow-hidden shadow-2xl border border-gray-700">
        <canvas ref="canvasRef" class="w-full h-full object-contain"></canvas>

        <div class="absolute top-2 left-2 bg-black/60 text-white px-3 py-1 rounded-lg text-sm font-medium">
          FPS: {{ frameRate }}
        </div>

        <div v-if="!isRunning" class="absolute inset-0 flex items-center justify-center text-gray-500">
          点击“开始识别”启动
        </div>
      </div>

      <!-- 右侧：Pinia 结算清单 -->
      <div class="w-96 bg-white rounded-xl shadow-lg p-6 flex flex-col border border-gray-200">
        <div class="flex justify-between items-center mb-4 border-b pb-4">
          <h2 class="text-xl font-black text-gray-800">结算清单</h2>
          <button @click="cartStore.clearCart" class="text-sm text-red-500 hover:text-red-700">清空</button>
        </div>

        <div class="flex-1 overflow-y-auto">
          <table class="w-full text-left border-collapse">
            <thead class="bg-gray-100 text-gray-600 sticky top-0">
              <tr>
                <th class="p-2 text-sm">商品名称</th>
                <th class="p-2 text-sm">单价</th>
                <th class="p-2 text-sm">数量</th>
              </tr>
            </thead>
            <tbody class="divide-y">
              <tr v-for="item in cartStore.currentFrameItems" :key="item.sku" class="hover:bg-gray-50">
                <td class="p-2 font-medium text-gray-800">{{ item.name }}</td>
                <td class="p-2 text-blue-600">￥{{ item.price.toFixed(2) }}</td>
                <td class="p-2 font-mono font-bold">{{ item.count }}</td>
              </tr>
              <tr v-if="cartStore.currentFrameItems.length === 0">
                <td colspan="3" class="text-center py-10 text-gray-400">当前视野内无商品</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="mt-4 pt-4 border-t border-gray-200">
          <div class="flex justify-between items-center mb-4">
            <span class="text-gray-600 font-medium">总计</span>
            <span class="text-3xl font-black text-blue-600">￥{{ cartStore.totalPrice.toFixed(2) }}</span>
          </div>
          <button
            class="w-full bg-blue-600 text-white py-4 rounded-xl font-bold text-lg hover:bg-blue-700 transition-all shadow-md"
          >
            确认结算
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue'
import axios from 'axios'
import { useCartStore } from '../store/cart'

const cartStore = useCartStore();
const mode = ref('camera');
const isRunning = ref(false);
const isPaused = ref(false);
const videoFileName = ref('');
const uploadStatus = ref(''); // 新增：上传状态
const canvasRef = ref(null);
const fileInputRef = ref(null);
const frameRate = ref(0);
const frameTimes = ref([]); // 存储最近几帧的时间戳
const maxFrameHistory = 10; // 计算平均帧率的帧数历史

let lastFrameTime = performance.now();
const ipCamUrl = ref('http://10.144.144.4:8080/video'); 

let ws = null;

// 用于释放旧内存的变量
let lastObjectURL = null;

const colorMap = {
  bagged: ['rgb(255, 200, 0)', 'rgb(255, 255, 255)'],
  bottled: ['rgb(200, 200, 200)', 'rgb(60, 122, 86)'],
  boxed: ['rgb(0, 255, 200)', 'rgb(64, 64, 64)'],
  canned: ['rgb(255, 0, 0)', 'rgb(255, 255, 255)'],
};

const handleFileUpload = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  uploadStatus.value = '文件上传中...';
  const formData = new FormData();
  formData.append('video', file);
  try {
    const res = await axios.post('/api/videos/upload_test', formData);
    videoFileName.value = res.data.filename;
    uploadStatus.value = '上传完成';
  } catch (error) {
    uploadStatus.value = '上传失败';
    console.error('Upload failed:', error);
  }
};

const stopAndClear = () => {
  if (ws) ws.close();
  isRunning.value = false;
  isPaused.value = false;
  videoFileName.value = '';
  uploadStatus.value = '';
  cartStore.clearCart();
  const canvas = canvasRef.value;
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);  // 释放内存
  if (lastObjectURL) URL.revokeObjectURL(lastObjectURL);
};
const clearVideoSource = () => {
  videoFileName.value = '';
  uploadStatus.value = '';
  // 重置文件输入框
  if (fileInputRef.value) {
    fileInputRef.value.value = '';
  }
  stopAndClear();
};

const toggleRecognition = () => {
  if (!isRunning.value) {
    isRunning.value = true;
    isPaused.value = false;
    initWebSocket();
  } else {
    isPaused.value = !isPaused.value;
    // 发送暂停/恢复消息给后端
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(isPaused.value ? 'pause' : 'resume');
    }
  }
};

const initWebSocket = () => {
  let source = '0';
  if (mode.value === 'file') {
    source = videoFileName.value;
  } else if (mode.value === 'ipcam') {
    source = encodeURIComponent(ipCamUrl.value);
  }
  // const source = mode.value === 'camera' ? '0' : videoFileName.value;
  // 连接到后端服务器的 WebSocket
  ws = new WebSocket(`ws://localhost:8000/ws/recognition?video=${source}`);
  
  // 必须设置为二进制类型
  ws.binaryType = 'arraybuffer';

  const canvas = canvasRef.value;
  const ctx = canvas.getContext('2d');

  ws.onmessage = async (event) => {
    try {
      const buffer = event.data;
      // 1. 解析 Header (前4字节) 获取 JSON 长度
      const view = new DataView(buffer);
      const jsonLen = view.getUint32(0);

      // 2. 提取 JSON 内容
      const jsonStr = new TextDecoder().decode(buffer.slice(4, 4 + jsonLen));
      const { results } = JSON.parse(jsonStr);

      // 3. 提取图片字节并创建 Blob URL
      const imgBytes = buffer.slice(4 + jsonLen);
      const blob = new Blob([imgBytes], { type: 'image/jpeg' });
      
      // 释放上一次的内存，防止内存泄漏
      if (lastObjectURL) URL.revokeObjectURL(lastObjectURL);
      const url = URL.createObjectURL(blob);
      lastObjectURL = url;

      // 4. 渲染
      const now = performance.now();
      frameTimes.value.push(now);
      
      // 保持最近 maxFrameHistory 帧的时间戳
      if (frameTimes.value.length > maxFrameHistory) {
        frameTimes.value.shift();
      }
      
      // 计算平均帧率
      if (frameTimes.value.length >= 2) {
        const deltas = [];
        for (let i = 1; i < frameTimes.value.length; i++) {
          deltas.push(frameTimes.value[i] - frameTimes.value[i - 1]);
        }
        const avgDelta = deltas.reduce((a, b) => a + b, 0) / deltas.length;
        if (avgDelta > 0) {
          frameRate.value = Math.round(1000 / avgDelta);
        }
      }
      
      lastFrameTime = now;

      const offscreenImg = new Image();
      offscreenImg.onload = () => {
        canvas.width = offscreenImg.width;
        canvas.height = offscreenImg.height;
        ctx.drawImage(offscreenImg, 0, 0);
        
        // 渲染检测框
        results.forEach(res => drawBox(ctx, res));
        cartStore.updateFromFrame(results);
      };
      offscreenImg.src = url;

    } catch (e) {
      console.error("解析二进制流失败:", e);
    }
  };

  ws.onclose = () => {
    isRunning.value = false;
    isPaused.value = false;
    ws = null;
  };

  ws.onerror = (err) => {
    console.error('WebSocket 错误:', err);
  };
};

const drawBox = (ctx, res) => {
  const [x1, y1, x2, y2] = res.bbox;
  const colors = colorMap[res.big_category] || ['white', 'black'];

  ctx.fillStyle = colors[0].replace('rgb', 'rgba').replace(')', ', 0.4)');
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = colors[0];
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  const label = `${res.product_name} ${res.score.toFixed(2)}`;
  ctx.font = 'bold 16px Arial';
  const textW = ctx.measureText(label).width;
  const labelH = 24;
  const yLabel = Math.max(0, y1 - labelH);

  ctx.fillStyle = colors[0];
  ctx.fillRect(x1, yLabel, textW + 10, labelH);
  ctx.fillStyle = colors[1];
  ctx.fillText(label, x1 + 5, yLabel + 18);
};

onUnmounted(stopAndClear);
</script>·