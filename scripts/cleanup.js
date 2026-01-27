#!/usr/bin/env node
/**
 * 清理脚本 - 停止所有相关进程并清理 PID 文件
 */
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PID_FILE = path.join(__dirname, '..', '.backend.pid');
const PORT = process.env.VITE_API_PORT || '8003';

console.log('🧹 开始清理...\n');

// 清理 PID 文件中的进程
if (fs.existsSync(PID_FILE)) {
  const pid = parseInt(fs.readFileSync(PID_FILE, 'utf8'));
  console.log(`📌 清理 PID 文件中的进程: ${pid}`);
  try {
    if (os.platform() === 'win32') {
      execSync(`taskkill /PID ${pid} /F /T`, { stdio: 'ignore' });
    } else {
      process.kill(pid, 'SIGKILL');
    }
    console.log(`   ✅ 已停止进程 ${pid}`);
  } catch (err) {
    console.log(`   ℹ️  进程 ${pid} 可能已不存在`);
  }
  fs.unlinkSync(PID_FILE);
}

// 清理占用端口的进程
console.log(`\n📌 检查端口 ${PORT} 占用情况...`);
try {
  let cmd, parser;
  if (os.platform() === 'win32') {
    cmd = `netstat -ano | findstr :${PORT}`;
    const output = execSync(cmd, { encoding: 'utf8' });
    const pids = new Set(
      output.split('\n')
        .map(line => line.trim().split(/\s+/).pop())
        .filter(pid => pid && /^\d+$/.test(pid))
    );

    pids.forEach(pid => {
      try {
        execSync(`taskkill /PID ${pid} /F /T`, { stdio: 'ignore' });
        console.log(`   ✅ 已停止进程 ${pid} (端口 ${PORT})`);
      } catch (err) {
        console.log(`   ℹ️  无法停止进程 ${pid}`);
      }
    });
  } else {
    cmd = `lsof -t -i:${PORT}`;
    const output = execSync(cmd, { encoding: 'utf8' });
    const pids = output.trim().split('\n').filter(Boolean);

    pids.forEach(pid => {
      try {
        process.kill(parseInt(pid), 'SIGKILL');
        console.log(`   ✅ 已停止进程 ${pid} (端口 ${PORT})`);
      } catch (err) {
        console.log(`   ℹ️  无法停止进程 ${pid}`);
      }
    });
  }
} catch (err) {
  console.log(`   ℹ️  端口 ${PORT} 未被占用`);
}

// 清理 Python 相关进程（可选）
console.log('\n📌 清理 llm_simulator 相关进程...');
try {
  if (os.platform() === 'win32') {
    const output = execSync('tasklist', { encoding: 'utf8' });
    const lines = output.split('\n').filter(line =>
      line.includes('python') || line.includes('uvicorn')
    );

    lines.forEach(line => {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 2) {
        const pid = parts[1];
        try {
          // 检查是否是 llm_simulator 进程
          const cmdline = execSync(`wmic process where ProcessId=${pid} get CommandLine`, {
            encoding: 'utf8'
          });
          if (cmdline.includes('llm_simulator')) {
            execSync(`taskkill /PID ${pid} /F /T`, { stdio: 'ignore' });
            console.log(`   ✅ 已停止 llm_simulator 进程 ${pid}`);
          }
        } catch (err) {
          // 忽略错误
        }
      }
    });
  } else {
    execSync("ps aux | grep '[l]lm_simulator' | awk '{print $2}' | xargs -r kill -9", {
      stdio: 'ignore'
    });
    console.log('   ✅ 已清理所有 llm_simulator 进程');
  }
} catch (err) {
  console.log('   ℹ️  没有找到 llm_simulator 进程');
}

console.log('\n✅ 清理完成！\n');
