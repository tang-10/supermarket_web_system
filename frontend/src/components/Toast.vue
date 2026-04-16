<template>
  <Teleport to="body">
    <div v-if="visible" class="fixed top-4 left-1/2 transform -translate-x-1/2 z-50">
      <div class="bg-white border rounded-lg shadow-lg p-4 min-w-96 max-w-md"
           :class="typeClasses">
        <div class="flex items-center">
          <div class="flex-1">
            <p class="font-medium">{{ message }}</p>
          </div>
          <button @click="close" class="ml-4 text-gray-400 hover:text-gray-600">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  message: {
    type: String,
    required: true
  },
  type: {
    type: String,
    default: 'info', // 'success', 'error', 'warning', 'info'
    validator: (value) => ['success', 'error', 'warning', 'info'].includes(value)
  },
  duration: {
    type: Number,
    default: 0 // 0表示不自动关闭
  }
})

const emit = defineEmits(['close'])

const visible = ref(false)

const typeClasses = {
  success: 'border-green-200 bg-green-50 text-green-800',
  error: 'border-red-200 bg-red-50 text-red-800',
  warning: 'border-yellow-200 bg-yellow-50 text-yellow-800',
  info: 'border-blue-200 bg-blue-50 text-blue-800'
}

const show = () => {
  visible.value = true
  if (props.duration > 0) {
    setTimeout(() => {
      close()
    }, props.duration)
  }
}

const close = () => {
  visible.value = false
  emit('close')
}

// 监听message变化时显示toast
watch(() => props.message, (newMessage) => {
  if (newMessage) {
    show()
  }
}, { immediate: true })

defineExpose({
  show,
  close
})
</script>