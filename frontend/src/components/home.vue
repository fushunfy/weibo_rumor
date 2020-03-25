<template>
  <div>
    <el-button class="filter-item" style="margin-left: 10px;" type="primary" @click="openCsvDialog()" icon="el-icon-plus">
      导入
    </el-button>
    <el-button class="filter-item" style="margin-left: 10px;" type="success" @click="runPredict()">
     运行
    </el-button>
    <el-dialog
      :title="csvTitle"
      :visible.sync="csvVisible"
      width="50%">
      <div>
        <el-form ref="file" label-width="120px">
          <el-form-item label="CSV文件导入：">
            <!-- auto-upload	是否在选取文件后立即进行上传-->
            <el-upload
              class="upload-demo"
              ref="upload"
              drag
              accept=".csv"
              action="resources/data/"
              :multiple="false"
              :limit="1"
              :auto-upload="false"
              :before-upload="beforeUpload"
              :on-change="handleChange">
              <i class="el-icon-upload"></i>
              <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
              <div class="el-upload__tip" slot="tip">只能上传csv文件</div>
            </el-upload>
          </el-form-item>
        </el-form>
        <el-table :data="tableData" border highlight-current-row style="width: 100%;margin-top:20px;">
          <el-table-column v-for="item of tableHeader" :key="item" :prop="item" :label="item" />
        </el-table>
      </div>
      <span slot="footer" class="dialog-footer">
    <el-button @click="csvVisible = false">取消</el-button>
    <el-button type="primary" @click="importCsv">导入</el-button>
    </span>
    </el-dialog>
  </div>

</template>

<script>
  import {uploadFile, runPredictFile} from '@/api/request'
  export default {
    name: "home",
    data() {
      return {
        filename: '',
        csvVisible: false,
        csvTitle: '',
        file: null,
        tableData: [],
        tableHeader: []
      }

    },
    methods:{
      //	上传文件之前的钩子，参数为上传的文件，若返回 false 或者返回 Promise 且被 reject，则停止上传。
      beforeUpload(file){
        // debugger
        this.file = file.raw;
        // const extension = file.name.split('.')[1] === 'xls'
        // const extension2 = file.name.split('.')[1] === 'xlsx'
        // const isLt2M = file.size / 1024 / 1024 < 5
        // if (!extension && !extension2) {
        //   this.$message.warning('上传模板只能是 xls、xlsx格式!')
        //   return
        // }
        // if (!isLt2M) {
        //   this.$message.warning('上传模板大小不能超过 5MB!')
        //   return
        // }
        this.filename = file.name;
        return false // 返回false不会自动上传
      },
      openCsvDialog() {
        this.file = null;
        this.filename = '';
        this.csvVisible = true;
        this.csvTitle = '导入CSV文件';
        if(this.$refs.upload){
          this.$refs.upload.clearFiles();
        }
      },
      // on-change	文件状态改变时的钩子，添加文件、上传成功和上传失败时都会被调用
      handleChange(file) {
        this.filename = file.name;
        this.file = file.raw;
      },
      async importCsv() {
        let fileFormData = new FormData();
        fileFormData.set(this.filename, this.file);
        if(this.filename !== ''){
          uploadFile(fileFormData, {headers: { "Content-type": "multipart/form-data" }}).then( res => {
            if (res.data.code === 0) {
              this.csvVisible = false;
              //这里是导入完文件后，重新查询数据库刷新页面
              this.$message({
                type: 'success',
                message: '导入成功',
                duration: 1500,
                onClose: async () => {
                }
              })
            }
          })
        }else{
          this.$message.error('上传文件不能为空');
        }
      },
      runPredict() {
        runPredictFile().then( res => {
          if(res.data.code === 0){
            this.$message({
              type: 'success',
              message: '开始运行',
              duration: 1500,
              onClose: async () => {
              }
            })
          }
        })
      }
    }


  }
</script>

<style scoped>

</style>































