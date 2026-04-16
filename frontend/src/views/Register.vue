<template>
  <div class="max-w-2xl mx-auto p-6 bg-white mt-2 rounded-2xl shadow-sm border border-gray-200">
    <h1 class="text-xl font-black text-gray-800 mb-4">新商品视频注册</h1>
    
    <div class="space-y-4">
      <div>
        <label class="block text-xs font-bold text-gray-700 mb-1">商品名称</label>
        <input v-model="form.product_name" type="text" class="w-full border-2 rounded-lg p-2.5 outline-none focus:border-blue-500 transition-colors text-sm" placeholder="如：可口可乐 330ml">
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-bold text-gray-700 mb-2">SKU 编码 (唯一)</label>
          <input v-model="form.sku" type="text" class="w-full border-2 rounded-lg p-3 outline-none focus:border-blue-500" placeholder="如：coke_330">
        </div>
        <div>
          <label class="block text-sm font-bold text-gray-700 mb-2">价格 (元)</label>
          <input v-model.number="form.price" type="number" class="w-full border-2 rounded-lg p-3 outline-none focus:border-blue-500" placeholder="0.00">
        </div>
      </div>

      <!-- 文件上传区域 -->
      <div class="border-2 border-dashed border-gray-300 bg-gray-50 rounded-xl p-6 text-center hover:border-blue-400 transition-colors relative group">
        <input type="file" @change="handleFile" class="absolute inset-0 opacity-0 cursor-pointer" accept="video/mp4,video/avi" />
        <div v-if="!file" class="text-gray-500 group-hover:text-blue-500 transition-colors">
          <p class="text-2xl mb-1">📤</p>
          <p class="font-medium">点击或拖拽商品全角度录屏视频到此处</p>
          <p class="text-xs mt-2 text-gray-400">支持 .mp4, .avi 格式</p>
        </div>
        <div v-else class="text-blue-600 font-bold text-lg">
            已选择: {{ file.name }}
        </div>
      </div>

      <!-- 提交按钮 -->
      <button @click="submit" :disabled="loading || polling" 
              class="w-full py-3 rounded-xl font-black text-lg transition-all shadow-md"
              :class="(loading || polling) ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-gray-900 text-white hover:bg-black active:scale-95'">
        {{ loading ? '正在上传...' : polling ? '后台处理中...' : '提交注册任务' }}
      </button>
    </div>

    <!-- Toast 通知 -->
    <Toast 
      v-if="toastMessage" 
      :message="toastMessage" 
      :type="toastType"
      @close="closeToast"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import Toast from '../components/Toast.vue'

const form = ref({ product_name: '', sku: '', price: '' })
const file = ref(null)
const loading = ref(false)
const polling = ref(false)
const toastMessage = ref('')
const toastType = ref('info')
let pollInterval = null

const handleFile = (e) => {
  file.value = e.target.files[0]
}

const showToast = (message, type = 'info') => {
  toastMessage.value = message
  toastType.value = type
}

const closeToast = () => {
  toastMessage.value = ''
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
    polling.value = false
  }
}

const pollTaskStatus = async (taskId) => {
  try {
    const response = await axios.get(`/api/registration/status/${taskId}`)
    const { status, message } = response.data

    if (status === 'uploading') {
      showToast('视频上传成功，等待后台提取特征...', 'info')
    } else if (status === 'processing') {
      showToast('正在提取特征...', 'warning')
    } else if (status === 'completed') {
      showToast('新商品注册成功，可以去收银台测试了。', 'success')
      // 清空表单
      file.value = null
      form.value = { product_name: '', sku: '', price: '' }
      // 停止轮询
      if (pollInterval) {
        clearInterval(pollInterval)
        pollInterval = null
        polling.value = false
      }
    } else if (status === 'failed') {
      showToast('新商品注册失败，请查询日志处理，或者重新上传注册视频。', 'error')
      // 停止轮询
      if (pollInterval) {
        clearInterval(pollInterval)
        pollInterval = null
        polling.value = false
      }
    }
  } catch (error) {
    console.error('轮询任务状态失败:', error)
    showToast('查询任务状态失败，请稍后手动刷新页面查看结果。', 'error')
    if (pollInterval) {
      clearInterval(pollInterval)
      pollInterval = null
      polling.value = false
    }
  }
}

const submit = async () => {
  if (!file.value || !form.value.sku || !form.value.product_name) {
    showToast('请填写完整信息并选择视频文件', 'error')
    return
  }
  
  loading.value = true
  
  const formData = new FormData()
  formData.append('video', file.value)
  formData.append('sku', form.value.sku)
  formData.append('product_name', form.value.product_name)
  formData.append('price', form.value.price)

  try {
    // 调用 FastAPI 的注册接口
    const res = await axios.post('/api/registration/upload', formData)
    const { task_id } = res.data
    
    // 开始轮询任务状态
    polling.value = true
    pollTaskStatus(task_id) // 立即查询一次
    pollInterval = setInterval(() => pollTaskStatus(task_id), 2000) // 每2秒轮询一次
    
  } catch (e) {
    showToast("注册失败: " + (e.response?.data?.detail || e.message), 'error')
  } finally {
    loading.value = false
  }
}
</script>