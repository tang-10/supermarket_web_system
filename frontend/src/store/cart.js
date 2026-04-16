import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useCartStore = defineStore('cart', () => {
  const currentFrameItems = ref([])

  const totalPrice = computed(() => {
    return currentFrameItems.value.reduce((sum, item) => sum + item.price * item.count, 0)
  })

  function updateFromFrame(results) {
    const groups = {}
    results.forEach(res => {
      if (res.fine_class === 'unknown') return
      if (!groups[res.sku]) {
        groups[res.sku] = {
          sku: res.sku,
          name: res.product_name,
          price: res.price,
          count: 0,
        }
      }
      groups[res.sku].count++
    })
    currentFrameItems.value = Object.values(groups)
  }

  function clearCart() {
    currentFrameItems.value = []
  }

  return {
    currentFrameItems,
    totalPrice,
    updateFromFrame,
    clearCart,
  }
})