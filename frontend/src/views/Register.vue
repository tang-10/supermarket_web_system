<template>
  <div class="max-w-2xl mx-auto p-8 bg-white mt-10 rounded-2xl shadow-sm border border-gray-200">
    <h1 class="text-2xl font-black text-gray-800 mb-8">📦 新商品视频注册</h1>
    
    <div class="space-y-6">
      <div>
        <label class="block text-sm font-bold text-gray-700 mb-2">商品名称</label>
        <input v-model="form.product_name" type="text" class="w-full border-2 rounded-lg p-3 outline-none focus:border-blue-500 transition-colors" placeholder="如：可口可乐 330ml">
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
      <div class="border-2 border-dashed border-gray-300 bg-gray-50 rounded-xl p-10 text-center hover:border-blue-400 transition-colors relative group">
        <input type="file" @change="handleFile" class="absolute inset-0 opacity-0 cursor-pointer" accept="video/mp4,video/avi" />
        <div v-if="!file" class="text-gray-500 group-hover:text-blue-500 transition-colors">
          <p class="text-4xl mb-3">📤</p>
          <p class="font-medium">点击或拖拽商品全角度录屏视频到此处</p>
          <p class="text-xs mt-2 text-gray-400">支持 .mp4, .avi 格式</p>
        </div>
        <div v-else class="text-blue-600 font-bold text-lg">
          📄 已选择: {{ file.name }}
        </div>
      </div>

      <!-- 提交按钮 -->
      <button @click="submit" :disabled="loading" 
              class="w-full py-4 rounded-xl font-black text-lg transition-all shadow-md"
              :class="loading ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-gray-900 text-white hover:bg-black active:scale-95'">
        {{ loading ? '⏳ 正在上传...' : '🚀 提交注册任务' }}
      </button>

      <!-- 异步反馈提示 -->
      <div v-if="message" class="p-4 rounded-lg font-medium" 
           :class="isSuccess ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'">
        {{ message }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const form = ref({ product_name: '', sku: '', price: '' })
const file = ref(null)
const loading = ref(false)
const message = ref('')
const isSuccess = ref(true)

const handleFile = (e) => {
  file.value = e.target.files[0]
}

const submit = async () => {
  if (!file.value || !form.value.sku || !form.value.product_name) {
    isSuccess.value = false
    message.value = '⚠️ 请填写完整信息并选择视频文件'
    return
  }
  
  loading.value = true
  message.value = ''
  
  const formData = new FormData()
  formData.append('video', file.value)
  formData.append('sku', form.value.sku)
  formData.append('product_name', form.value.product_name)
  formData.append('price', form.value.price)

  try {
    // 调用 FastAPI 的注册接口
    const res = await axios.post('/api/registration/upload', formData)
    isSuccess.value = true
    message.value = "✅ 视频上传成功！后台 AI 正在疯狂提取特征中... 您可以去收银台测试了。"
    // 清空表单
    file.value = null
    form.value = { product_name: '', sku: '', price: '' }
  } catch (e) {
    isSuccess.value = false
    message.value = "❌ 注册失败: " + (e.response?.data?.detail || e.message)
  } finally {
    loading.value = false
  }
}
</script>