import { createRouter, createWebHistory } from 'vue-router'
import Monitor from '../views/Monitor.vue'
import Register from '../views/Register.vue'

const routes =[
  { path: '/', redirect: '/monitor' },
  { path: '/monitor', component: Monitor },
  { path: '/register', component: Register }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router