import request from '@/utils/request'

export function login(data) {
  return request({
    url: '/',
    method: 'post',
    data
  })
}

export function register(data) {
  return request({
    url: '/register',
    method: 'post',
    data
  })
}

export function uploadFile(data) {
  return request({
    url: '/home/uploadFile',
    method: 'post',
    data
  })
}

export function runPredictFile() {
  return request({
    url: '/home/runPredictFile',
    method: 'get'
  })
}
