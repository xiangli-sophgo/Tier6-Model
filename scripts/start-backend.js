#!/usr/bin/env node
/**
 * 后端启动脚本 - 带进程锁和自动清理
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PID_FILE = path.join(__dirname, '..', '.backend.pid');
const PORT = process.env.VITE_API_PORT;
if (!PORT) {
  console.error('[ERROR] VITE_API_PORT is not set. Please create .env file with VITE_API_PORT=<port>');
  process.exit(1);
}

// 清理函数
function cleanup() {
  if (fs.existsSync(PID_FILE)) {
    const pid = parseInt(fs.readFileSync(PID_FILE, 'utf8'));
    console.log(`🧹 清理旧进程 PID: ${pid}`);
    try {
      if (os.platform() === 'win32') {
        spawn('taskkill', ['/PID', pid.toString(), '/F', '/T'], { stdio: 'ignore' });
      } else {
        process.kill(pid, 'SIGTERM');
      }
    } catch (err) {
      // 进程可能已经不存在了
    }
    fs.unlinkSync(PID_FILE);
  }
}

// 检查端口占用
function checkPort() {
  return new Promise((resolve) => {
    const cmd = os.platform() === 'win32'
      ? `netstat -ano | findstr :${PORT}`
      : `lsof -i :${PORT}`;

    require('child_process').exec(cmd, (err, stdout) => {
      if (stdout && stdout.trim()) {
        console.log(`⚠️  端口 ${PORT} 已被占用，正在清理...`);
        cleanup();
        setTimeout(resolve, 2000);
      } else {
        resolve();
      }
    });
  });
}

// 启动后端
async function start() {
  console.log('🚀 启动后端服务...');

  // 清理旧进程
  await checkPort();
  cleanup();

  // 启动新进程
  const python = process.platform === 'win32' ? 'python' : 'python3';
  const backend = spawn(python, ['-m', 'math_model.main'], {
    cwd: path.join(__dirname, '..', 'backend'),
    stdio: 'inherit',
    env: { ...process.env }
  });

  // 保存 PID
  fs.writeFileSync(PID_FILE, backend.pid.toString());
  console.log(`✅ 后端服务已启动 (PID: ${backend.pid}, Port: ${PORT})`);

  // 进程退出时清理
  backend.on('exit', (code) => {
    console.log(`🛑 后端服务已停止 (退出码: ${code})`);
    cleanup();
  });

  // 处理 Ctrl+C
  process.on('SIGINT', () => {
    console.log('\n🛑 收到停止信号，正在清理...');
    backend.kill('SIGTERM');
    cleanup();
    process.exit(0);
  });
}

start().catch(err => {
  console.error('❌ 启动失败:', err);
  cleanup();
  process.exit(1);
});
