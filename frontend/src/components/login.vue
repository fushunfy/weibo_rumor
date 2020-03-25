<template>
  <div class="login">
    <!--    <div></div>-->
    <el-form ref="loginForm" :model="loginForm" :rules="loginRules" class="login-form" auto-complete="on"
             label-position="left">
      <div class="title-container">
        <h2 class="title">登录</h2>
      </div>

      <el-form-item prop="username">
        <el-input
          ref="username"
          v-model="loginForm.username"
          placeholder="Username"
          name="username"
          type="text"
          tabindex="1"
          auto-complete="on"
          prefix-icon="el-icon-user"
        />
      </el-form-item>
      <el-form-item prop="password">
        <el-input
          ref="password"
          v-model="loginForm.password"
          placeholder="Password"
          name="password"
          tabindex="2"
          auto-complete="on"
          @keyup.enter.native="handleLogin"
          show-password
          prefix-icon="el-icon-key"
        />
      </el-form-item>

      <el-button :loading="loading" type="primary" style="width:100%;margin-bottom:20px;"
                 @click.native.prevent="handleLogin">Login
      </el-button>
      <div class="login_control">
        <router-link :to="'/register'" class="go_register" style="font-size: small">
          <span>还没有账户? 立即注册</span>
        </router-link>
        <router-link :to="'/findpassword'" class="findpassword"  style="font-size: 12px">
          <span>忘记密码</span>
        </router-link>
      </div>
    </el-form>

  </div>
</template>

<script>
  import {login} from '@/api/request'
  import { validUsername, validPassword } from '@/utils/validate'

  export default {
    name: 'login',
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
        loginForm: {
          username: '',
          password: ''
        },
        // blur失去焦点
        loginRules: {
          username: [{required: true, trigger: 'blur', validator: validateUsername}],
          password: [{required: true, trigger: 'blur', validator: validatePassword}]
        },
        loading: false
      }
    },
    methods: {
      handleLogin() {
        this.$refs.loginForm.validate(valid => {
          if (valid) {
            this.loading = true
            login(this.loginForm).then(res => {
              if (res.data.code === 0) {
                this.$router.push({path: "/home"});
                this.loading = false
              } else{
                var mes = res.data.message
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
  .login {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-image: url(../assets/images/bk_one.jpg);
    background-size: 100%;
    background-repeat: no-repeat;
  }

  .login-form {
    width: 400px;
    height: 280px;
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

  /*.login-box-msg,*/
  /*.register-box-msg {*/
  /*  margin: 0;*/
  /*  text-align: center;*/
  /*  padding: 0 20px 20px 20px;*/
  /*  font-size: 24px;*/
  /*}*/

  .text-red {
    color: #dd4b39
  }

  /*.checkbox {*/
  /*  margin-left: -170px;*/
  /*}*/

  .go_register {
    margin-left: -5px;
  }

  .findpassword {
    margin-left: 215px;
  }
  .login_control {
    margin-left: -2px;
  }
</style>
<!--<style lang="scss">-->
<!--  /* 修复input 背景不协调 和光标变色 */-->
<!--  /* Detail see https://github.com/PanJiaChen/vue-element-admin/pull/927 */-->

<!--  $bg:#283443;-->
<!--  $light_gray:#fff;-->
<!--  $cursor: #fff;-->

<!--  @supports (-webkit-mask: none) and (not (cater-color: $cursor)) {-->
<!--    .login-container .el-input input {-->
<!--      color: $cursor;-->
<!--    }-->
<!--  }-->

<!--  /* reset element-ui css */-->
<!--  .login-container {-->
<!--    .el-input {-->
<!--      display: inline-block;-->
<!--      height: 47px;-->
<!--      width: 85%;-->

<!--      input {-->
<!--        background: transparent;-->
<!--        border: 0px;-->
<!--        -webkit-appearance: none;-->
<!--        border-radius: 0px;-->
<!--        padding: 12px 5px 12px 15px;-->
<!--        color: $light_gray;-->
<!--        height: 47px;-->
<!--        caret-color: $cursor;-->

<!--        &:-webkit-autofill {-->
<!--          box-shadow: 0 0 0px 1000px $bg inset !important;-->
<!--          -webkit-text-fill-color: $cursor !important;-->
<!--        }-->
<!--      }-->
<!--    }-->

<!--    .el-form-item {-->
<!--      border: 1px solid rgba(255, 255, 255, 0.1);-->
<!--      background: rgba(0, 0, 0, 0.1);-->
<!--      border-radius: 5px;-->
<!--      color: #454545;-->
<!--    }-->
<!--  }-->
<!--</style>-->
<!--&lt;!&ndash;当一个style标签拥有scoped属性时，它的CSS样式就只能作用于当前的组件，也就是说，该样式只能适用于当前组件元素。-->
<!--通过该属性，可以使得组件之间的样式不互相污染。如果一个项目中的所有style标签全部加上了scoped，相当于实现了样式的模块化。&ndash;&gt;-->
<!--<style lang="scss" scoped>-->
<!--  $bg:#2d3a4b;-->
<!--  $dark_gray:#889aa4;-->
<!--  $light_gray:#eee;-->

<!--  .login-container {-->
<!--    min-height: 100%;-->
<!--    width: 100%;-->
<!--    background-color: $bg;-->
<!--    overflow: hidden;-->

<!--    .login-form {-->
<!--      position: relative;-->
<!--      width: 520px;-->
<!--      max-width: 100%;-->
<!--      padding: 160px 35px 0;-->
<!--      margin: 0 auto;-->
<!--      overflow: hidden;-->
<!--    }-->

<!--    .tips {-->
<!--      font-size: 14px;-->
<!--      color: #fff;-->
<!--      margin-bottom: 10px;-->

<!--      span {-->
<!--        &:first-of-type {-->
<!--          margin-right: 16px;-->
<!--        }-->
<!--      }-->
<!--    }-->

<!--    .svg-container {-->
<!--      padding: 6px 5px 6px 15px;-->
<!--      color: $dark_gray;-->
<!--      vertical-align: middle;-->
<!--      width: 30px;-->
<!--      display: inline-block;-->
<!--    }-->

<!--    .title-container {-->
<!--      position: relative;-->

<!--      .title {-->
<!--        font-size: 26px;-->
<!--        color: $light_gray;-->
<!--        margin: 0px auto 40px auto;-->
<!--        text-align: center;-->
<!--        font-weight: bold;-->
<!--      }-->
<!--    }-->

<!--    .show-pwd {-->
<!--      position: absolute;-->
<!--      right: 10px;-->
<!--      top: 7px;-->
<!--      font-size: 16px;-->
<!--      color: $dark_gray;-->
<!--      cursor: pointer;-->
<!--      user-select: none;-->
<!--    }-->
<!--  }-->
<!--</style>-->
