<template>
  <div class="min-h-screen flex flex-col">
    <!-- 顶部导航栏 -->
    <header class="bg-gray-900 text-white shadow-md">
      <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <h1 class="text-xl font-bold tracking-wider text-blue-400">商超结算系统</h1>
        <nav class="space-x-4">
          <router-link to="/monitor" class="hover:text-blue-400 px-3 py-2 rounded-md transition-colors" active-class="bg-gray-800 text-blue-400">实时收银</router-link>
          <router-link to="/register" class="hover:text-blue-400 px-3 py-2 rounded-md transition-colors" active-class="bg-gray-800 text-blue-400">商品注册</router-link>
        </nav>
      </div>
    </header>

    <!-- 路由页面展示区 -->
    <main class="flex-1">
      <router-view></router-view>
    </main>
  </div>
</template>