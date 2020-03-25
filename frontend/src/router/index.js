import Vue from 'vue'
import Router from 'vue-router'
import login from '@/components/login'
import home from '@/components/home'
import register from '@/components/register'
import findpassword from '@/components/findpassword'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'login',
      component: login,
      meta: {
        keeplive: false
      }
    },
    {
      path: '/home',
      name: 'home',
      component: home,
      meta: {
        keeplive: false
      }
    },
    {
      path: '/register',
      name: 'register',
      component: register,
      meta: {
        keeplive: false
      }
    },
    {
      path: '/findpassword',
      name: 'findpassword',
      component: findpassword,
      meta: {
        keeplive: false
      }
    }
  ]
})
