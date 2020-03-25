<template>
  <div class="app-container">
    <div class="filter-container" style="margin-bottom:10px">
      <el-input
        v-model="listQuery.taskName"
        placeholder="任务名称"
        style="width: 150px;"
        class="filter-item"
        size="small"
        @keyup.enter.native="handleFilter"
      />
      <el-select
        v-model="listQuery.svrType"
        placeholder="服务类型"
        clearable
        class="filter-item"
        style="width: 130px"
        size="small"
      >
        <el-option v-for="(value,key) in svrTypeList" :key="key" :label="value" :value="value" />
      </el-select>
      <el-select
        v-model="listQuery.sort"
        style="width: 140px"
        class="filter-item"
        size="small"
        @change="handleFilter"
      >
        <el-option
          v-for="item in sortOptions"
          :key="item.key"
          :label="item.label"
          :value="item.key"
          size="small"
        />
      </el-select>
      <el-button
        v-waves
        class="filter-item"
        type="primary"
        icon="el-icon-search"
        size="small"
        @click="handleFilter"
      >搜索
      </el-button>
      <el-button
        class="filter-item"
        style="margin-left: 10px;"
        type="primary"
        icon="el-icon-edit"
        size="small"
        @click="handleCreate"
      >新增任务
      </el-button>
    </div>

    <el-table
      :key="tableKey"
      v-loading="listLoading"
      :data="list"
      border
      highlight-current-row
      style="width: 100%"
      :header-cell-style="{padding: '7px', 'text-align':'center'}"
      :cell-style="{padding: '5px', 'text-align':'center'}"
      :fit="true"
      @sort-change="sortChange"
    >
      <el-table-column label="任务ID" prop="taskId" width="90">
        <template slot-scope="scope">
          <span>{{ scope.row.taskId }}</span>
        </template>
      </el-table-column>
      <el-table-column
        label="任务名称"
        prop="taskName"
        min-width="150px"
      >
        <template slot-scope="scope">
          <router-link :to="'/task/detail/'+scope.row.taskId" class="link-type">
            <span>{{ scope.row.taskName }}</span>
          </router-link>
        </template>
      </el-table-column>
      <el-table-column label="服务类型" prop="svrType" width="140">
        <template slot-scope="scope">
          <span>{{ scope.row.svrName }}</span>
        </template>
      </el-table-column>
      <el-table-column label="环境" prop="env" width="100">
        <template slot-scope="scope">
          <span>{{ scope.row.envName }}</span>
        </template>
      </el-table-column>
      <el-table-column label="创建时间" width="170px">
        <template slot-scope="scope">
          <span>{{ scope.row.createTime | parseTime('{y}-{m}-{d} {h}:{i}:{s}') }}</span>
        </template>
      </el-table-column>
      <el-table-column label="任务状态" prop="taskStatus" class-name="status-col" width="100">
        <template slot-scope="scope">
          <el-tag
            size="small"
            :type="scope.row.taskStatus | statusFilter"
          >{{ statusOptions[scope.row.taskStatus] }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column width="90" class-name="small-padding fixed-width">
        <template slot="header">
          <span>订阅</span>
          <el-tooltip
            effect="dark"
            content="已订阅的任务可在[我的订阅]页面查看"
            placement="top"
          >
            <span><i class="el-icon-info" /></span>
          </el-tooltip>
        </template>
        <!--          当前行数据的获取也会用到插槽,scope相当于一行的数据， scope.row相当于当前行的数据对象-->
        <template slot-scope="scope">
          <el-button
            v-if="scope.row.subscribe==0"
            size="mini"
            type="success"
            @click="handleModifySubcribe(scope.row,1)"
          >订阅
          </el-button>
          <el-button
            v-if="scope.row.subscribe==1"
            size="mini"
            @click="handleModifySubcribe(scope.row,0)"
          >取消
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-show="total>0"
      :total="total"
      :page.sync="listQuery.page"
      :limit.sync="listQuery.limit"
      @pagination="getList"
    />

    <el-dialog :title="textMap[dialogStatus]" :visible.sync="dialogFormVisible" :close-on-click-modal="false">
      <el-form
        ref="dataForm"
        :rules="rules"
        :model="temp"
        label-position="left"
        label-width="100px"
        style="width: 400px; margin-left:50px;"
      >
        <el-form-item label="任务名称" prop="taskName">
          <el-input v-model="temp.taskName" size="small" />
        </el-form-item>
        <el-form-item label="服务类型" prop="svrType">
          <el-select
            v-model="temp.svrType"
            class="filter-item"
            placeholder="请选择服务类型"
            size="small"
            @change="selectSvr($event)"
          >
            <el-option v-for="(value,key) in svrTypeList" :key="key" :label="value" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item label="环境" prop="env">
          <el-select v-model="temp.env" class="filter-item" placeholder="请选择压测环境" size="small">
            <el-option v-for="(value,key) in envList" :key="key" :label="value" :value="key" />
          </el-select>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button size="small" @click="dialogFormVisible = false">取消</el-button>
        <el-button type="primary" size="small" @click="dialogStatus==='create'?createData():updateData()">确认</el-button>
      </div>
    </el-dialog>

    <el-dialog :visible.sync="dialogPvVisible" title="Reading statistics">
      <el-table :data="pvData" border fit highlight-current-row style="width: 100%">
        <el-table-column prop="key" label="Channel" />
        <el-table-column prop="pv" label="Pv" />
      </el-table>
      <span slot="footer" class="dialog-footer">
        <el-button type="primary" @click="dialogPvVisible = false">确认</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script>
  import { fetchList, createTask, addSub, delSub } from '@/api/task'
  import { getSvrAll, getSvrEnvList } from '@/api/svr'
  import waves from '@/directive/waves' // Waves  directive
  import { parseTime } from '@/utils'
  import Pagination from '@/components/Pagination' // Secondary package based on el-pagination

  export default {
    name: 'TaskList',
    components: { Pagination },
    directives: { waves },
    filters: {
      statusFilter(status) {
        const statusMap = {
          0: 'success',
          1: 'danger'
        }
        return statusMap[status]
      },
      typeFilter(type) {
        return this.svrTypeKeyValue[type]
      }
    },
    data() {
      return {
        tableKey: 0,
        list: null,
        total: 0,
        listLoading: true,
        listQuery: {
          page: 1,
          limit: 10,
          taskName: undefined,
          svrType: undefined,
          sort: '+taskId'
        },
        svrTypeList: {},
        envList: {},
        svrTypeKeyValue: null,
        subcribe: 1,
        sortOptions: [
          { label: 'ID Ascending', key: '+taskId' },
          { label: 'ID Descending', key: '-taskId' }
        ],
        statusOptions: ['空闲', '运行中'],
        showReviewer: false,
        temp: {
          taskName: '',
          taskStatus: 0,
          taskMode: 1,
          svrType: 0,
          env: 0
        },
        dialogFormVisible: false,
        dialogStatus: '',
        textMap: {
          update: '编辑任务',
          create: '新增任务'
        },
        dialogPvVisible: false,
        pvData: [],
        rules: {
          svrType: [
            { required: true, message: '服务类型必选', trigger: 'change' }
          ],
          env: [{ required: true, message: '压测环境必选', trigger: 'change' }],
          taskName: [
            { required: true, message: '请填写任务名称', trigger: 'blur' }
          ]
        },
        downloadLoading: false
      }
    },
    created() {
      // 服务列表
      getSvrAll().then(res => {
        res.data.forEach(element => {
          this.svrTypeList[element.id] = element.serviceName
        })
      })
      this.getList()
    },
    methods: {
      getList() {
        this.listLoading = true
        fetchList(this.listQuery).then(response => {
          this.list = response.data.items
          this.total = response.data.total
          this.listLoading = false
          // Just to simulate the time of the request
          // setTimeout(() => {
          //   this.listLoading = false;
          // }, 1.5 * 1000);
        })
      },
      handleFilter() {
        this.listQuery.page = 1
        this.getList()
      },
      // 任务状态
      handleTaskModifyStatus(row, status) {
        if (status == 1) {
          this.$confirm('点击确认开始压测任务', '提示', {
            confirmButtonText: '确认',
            cancelButtonText: '取消',
            type: 'warning'
          })
            .then(() => {
              this.$message({
                message: '操作成功',
                type: 'success'
              })
            })
            .catch()
        }
        row.taskStatus = status
      },
      // 订阅状态
      handleModifySubcribe(row, status) {
        if (status == 0) {
          delSub(row.taskId).then(res => {
            if (res.code == 0) {
              this.$message({
                message: '已取消',
                type: 'success'
              })
            } else {
              this.$message({
                message: res.msg,
                type: 'danger'
              })
            }
          })
        } else {
          addSub(row.taskId).then(res => {
            if (res.code == 0) {
              this.$message({
                message: '已订阅',
                type: 'success'
              })
            } else {
              this.$message({
                message: res.msg,
                type: 'danger'
              })
            }
          })
        }
        row.subscribe = status
      },
      sortChange(data) {
        const { prop, order } = data
        if (prop === 'id') {
          this.sortByID(order)
        }
      },
      sortByID(order) {
        if (order === 'ascending') {
          this.listQuery.sort = '+taskId'
        } else {
          this.listQuery.sort = '-taskId'
        }
        this.handleFilter()
      },
      resetTemp() {
        this.temp = {
          taskName: '',
          taskStatus: 0,
          taskMode: 1,
          svrType: undefined,
          env: undefined
        }
      },
      resetEnvOption() {
        this.envList = {}
        this.temp.env = undefined
      },
      handleCreate() {
        this.resetTemp()
        this.dialogStatus = 'create'
        this.dialogFormVisible = true
        this.$nextTick(() => {
          this.$refs['dataForm'].clearValidate()
        })
      },
      // 创建任务-服务选择事件
      selectSvr(svrType) {
        // 环境列表
        getSvrEnvList(svrType).then(res => {
          this.resetEnvOption()
          res.data.forEach(element => {
            this.envList[element.id] = element.configName
          })
        })
      },
      createData() {
        this.$refs['dataForm'].validate(valid => {
          if (valid) {
            createTask(this.temp).then(res => {
              this.list.unshift(res.data)
              this.dialogFormVisible = false
              this.$notify({
                title: '成功',
                message: '创建成功',
                type: 'success',
                duration: 2000
              })
            })
          }
        })
      },
      handleUpdate(row) {
        this.temp = Object.assign({}, row) // copy obj
        this.temp.timestamp = new Date(this.temp.timestamp)
        this.dialogStatus = 'update'
        this.dialogFormVisible = true
        this.$nextTick(() => {
          this.$refs['dataForm'].clearValidate()
        })
      },
      formatJson(filterVal, jsonData) {
        return jsonData.map(v =>
          filterVal.map(j => {
            if (j === 'timestamp') {
              return parseTime(v[j])
            } else {
              return v[j]
            }
          })
        )
      }
    }
  }
</script>

