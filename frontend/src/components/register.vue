<template>
  <div class="register">
    <!--    <div></div>-->
    <el-form ref="registerForm" :model="registerForm" :rules="registerRules" class="register-form" auto-complete="on"
             label-position="left">
      <div class="title-container">
        <h2 class="title">注册</h2>
      </div>

      <el-form-item prop="username">
        <el-input
          ref="username"
          v-model="registerForm.username"
          placeholder="Username"
          name="username"
          type="text"
          tabindex="1"
          auto-complete="on"
          prefix-icon="el-icon-user-solid"
        />
      </el-form-item>
      <el-form-item prop="email">
        <el-input
          ref="email"
          v-model="registerForm.email"
          placeholder="Email"
          name="email"
          tabindex="2"
          auto-complete="on"
          prefix-icon="el-icon-message"
        />
      </el-form-item>
      <el-form-item prop="password">
        <el-input
          ref="password"
          v-model="registerForm.password"
          placeholder="Password"
          name="password"
          tabindex="2"
          auto-complete="on"
          @keyup.enter.native="handleRegister"
          show-password
          prefix-icon="el-icon-key"
        />
      </el-form-item>

      <el-button :loading="loading" type="primary" style="width:100%;margin-bottom:20px;"
                 @click.native.prevent="handleRegister">Register
      </el-button>
    </el-form>

  </div>
</template>

<script>
  import {register} from '@/api/request'
  import { validUsername, validPassword } from '@/utils/validate'

  export default {
    name: 'register',
    data() {
      const validateUsername = (rule, value, callback) => {
        // trim 表示字符串去除字符串最左边和最右边的空格
        if (!validUsername(value)) {
          callback(new Error('Please enter the correct user name'))
        } else {
          callback()
        }
      };
      const validatePassword = (rule, value, callback) => {
        if (!validPassword(value)) {
          callback(new Error('The password can not be less than 6 digits'))
        } else {
          callback()
        }
      };
      return {
        registerForm: {
          username: '',
          email: '',
          password: ''
        },
        // blur失去焦点
        registerRules: {
          username: [{required: true, trigger: 'blur', validator: validateUsername}],
          email: [
            { required: true, message: '请输入邮箱地址', trigger: 'blur' },
            { type: 'email', message: '请输入正确的邮箱地址', trigger: ['blur', 'change'] }
            ],
          password: [{required: true, trigger: 'blur', validator: validatePassword}]
        },
        loading: false
      }
    },
    methods: {
      handleRegister() {
        this.$refs.registerForm.validate(valid => {
          if (valid) {
            this.loading = true;
            register(this.registerForm).then(res => {
              if (res.data.code === 0) {
                this.$router.push({path: "/"});
                this.loading = false
              } else{
                var mes = res.data.message;
                console.log(res, mes)
                this.$message(mes);
                this.loading = false
              }
            }).catch(() => {
              this.loading = false
            })
          }
        })
      }
    }
  }
</script>

<style scoped>
  .register {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-image: url(../assets/images/bk_one.jpg);
    background-size: 100%;
    background-repeat: no-repeat;
  }

  .register-form {
    width: 400px;
    height: 300px;
    padding: 13px;
    position: absolute;
    left: 50%;
    top: 50%;
    margin-left: -200px;
    margin-top: -200px;

    background-color: rgba(240, 255, 255, 0.5);

    border-radius: 10px;
    text-align: center;
  }
</style>
