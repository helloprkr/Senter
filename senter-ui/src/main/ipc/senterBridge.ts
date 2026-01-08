import { ipcMain, BrowserWindow } from 'electron'
import * as net from 'net'
import { showResearchComplete } from '../notifications'

const SOCKET_PATH = '/tmp/senter.sock'
const CONNECTION_TIMEOUT = 10000
const POLL_INTERVAL = 5000  // P2-005: Poll every 5 seconds for task completions

interface SenterResponse {
  success: boolean
  data?: unknown
  error?: string
}

class SenterClient {
  private async sendCommand(command: string, args: Record<string, unknown> = {}): Promise<SenterResponse> {
    console.log(`[SenterClient] Sending command: ${command}`, args)

    return new Promise((resolve) => {
      const client = new net.Socket()
      let response = ''

      const timeout = setTimeout(() => {
        console.log(`[SenterClient] Connection timeout for: ${command}`)
        client.destroy()
        resolve({ success: false, error: 'Connection timeout' })
      }, CONNECTION_TIMEOUT)

      client.connect(SOCKET_PATH, () => {
        console.log(`[SenterClient] Connected to ${SOCKET_PATH}`)
        const message = JSON.stringify({ command, ...args }) + '\n'
        console.log(`[SenterClient] Writing message: ${message.trim()}`)
        client.write(message)
      })

      client.on('data', (data) => {
        response += data.toString()
        console.log(`[SenterClient] Received data chunk, total length: ${response.length}`)
      })

      client.on('error', (err) => {
        console.log(`[SenterClient] Socket error:`, err.message)
        clearTimeout(timeout)
        resolve({ success: false, error: `Connection failed: ${err.message}` })
      })

      client.on('end', () => {
        clearTimeout(timeout)
        console.log(`[SenterClient] Connection ended, response length: ${response.length}`)
        if (response) {
          try {
            const parsed = JSON.parse(response.trim())
            console.log(`[SenterClient] Parsed response:`, parsed)
            resolve({ success: true, data: parsed })
          } catch (e) {
            console.log(`[SenterClient] Parse error:`, e)
            resolve({ success: false, error: 'Invalid response' })
          }
        } else {
          console.log(`[SenterClient] Connection ended without response`)
          resolve({ success: false, error: 'No response received' })
        }
      })

      client.on('close', () => {
        clearTimeout(timeout)
      })
    })
  }

  async query(text: string, conversationId?: string): Promise<SenterResponse> {
    return this.sendCommand('query', { text, conversation_id: conversationId })
  }

  async getResearch(limit = 10): Promise<SenterResponse> {
    return this.sendCommand('get_results', { limit })
  }

  async getStatus(): Promise<SenterResponse> {
    return this.sendCommand('status')
  }

  async getHealth(): Promise<SenterResponse> {
    return this.sendCommand('health')
  }

  async getConversations(conversationId?: string, includeMessages = false): Promise<SenterResponse> {
    return this.sendCommand('get_conversations', {
      conversation_id: conversationId,
      include_messages: includeMessages
    })
  }

  async saveConversation(conversation: {
    id?: string
    messages: Array<{ role: string; content: string }>
    focus?: string
    created?: string
  }): Promise<SenterResponse> {
    return this.sendCommand('save_conversation', { conversation })
  }

  async getTasks(limit = 20): Promise<SenterResponse> {
    return this.sendCommand('get_tasks', { limit })
  }

  async getJournal(date?: string, limit = 7): Promise<SenterResponse> {
    return this.sendCommand('get_journal', { date, limit })
  }

  async generateJournal(date?: string): Promise<SenterResponse> {
    return this.sendCommand('generate_journal', { date })
  }

  // P3-001: Context Sources
  async getContextSources(): Promise<SenterResponse> {
    return this.sendCommand('get_context_sources')
  }

  async addContextSource(source: {
    type: string
    title: string
    path: string
    description?: string
  }): Promise<SenterResponse> {
    return this.sendCommand('add_context_source', source)
  }

  async removeContextSource(id: string): Promise<SenterResponse> {
    return this.sendCommand('remove_context_source', { id })
  }

  // P4-001: Voice Input
  async startVoiceInput(): Promise<SenterResponse> {
    return this.sendCommand('start_voice_input')
  }

  async stopVoiceInput(): Promise<SenterResponse> {
    return this.sendCommand('stop_voice_input')
  }

  async getAudioStatus(): Promise<SenterResponse> {
    return this.sendCommand('get_audio_status')
  }

  // P4-002: Gaze Detection
  async getGazeStatus(): Promise<SenterResponse> {
    return this.sendCommand('get_gaze_status')
  }

  // V3-004: Get insights (patterns, preferences, stats)
  async getInsights(days = 7): Promise<SenterResponse> {
    return this.sendCommand('get_insights', { days })
  }

  // V3-005: Get goals
  async getGoals(status?: string, limit = 50): Promise<SenterResponse> {
    return this.sendCommand('get_goals', { status, limit })
  }

  // V3-005: Update goal status
  async updateGoal(goalId: string, status: string): Promise<SenterResponse> {
    return this.sendCommand('update_goal', { goal_id: goalId, status })
  }
}

const senterClient = new SenterClient()

// P2-005: Task completion poller for notifications
interface TaskInfo {
  id: string
  title: string
  status: string
  result?: string
}

class TaskPoller {
  private knownCompletedTasks: Set<string> = new Set()
  private pollInterval: ReturnType<typeof setInterval> | null = null
  private mainWindow: BrowserWindow | null = null
  private initialized = false

  setMainWindow(window: BrowserWindow): void {
    this.mainWindow = window
    console.log('[TaskPoller] Main window set')
  }

  start(): void {
    if (this.pollInterval) {
      console.log('[TaskPoller] Already running')
      return
    }

    console.log('[TaskPoller] Starting task completion polling')

    // Initialize with current completed tasks (don't notify for existing ones)
    this.initializeKnownTasks()

    this.pollInterval = setInterval(() => {
      this.checkForCompletions()
    }, POLL_INTERVAL)
  }

  stop(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval)
      this.pollInterval = null
      console.log('[TaskPoller] Stopped')
    }
  }

  private async initializeKnownTasks(): Promise<void> {
    try {
      const response = await senterClient.getTasks(50)
      if (response.success && response.data) {
        const data = response.data as { tasks?: TaskInfo[] }
        if (data.tasks) {
          data.tasks.forEach(task => {
            if (task.status === 'completed') {
              this.knownCompletedTasks.add(task.id)
            }
          })
          console.log(`[TaskPoller] Initialized with ${this.knownCompletedTasks.size} completed tasks`)
          this.initialized = true
        }
      }
    } catch (error) {
      console.error('[TaskPoller] Failed to initialize:', error)
    }
  }

  private async checkForCompletions(): Promise<void> {
    if (!this.initialized) {
      // Wait for initialization
      return
    }

    try {
      const response = await senterClient.getTasks(20)
      if (!response.success || !response.data) return

      const data = response.data as { tasks?: TaskInfo[] }
      if (!data.tasks) return

      // Find newly completed tasks
      const newlyCompleted = data.tasks.filter(task =>
        task.status === 'completed' && !this.knownCompletedTasks.has(task.id)
      )

      for (const task of newlyCompleted) {
        console.log(`[TaskPoller] New task completed: ${task.title}`)
        this.knownCompletedTasks.add(task.id)

        // Show desktop notification
        const resultPreview = task.result
          ? (task.result.length > 150 ? task.result.substring(0, 150) + '...' : task.result)
          : 'Research completed'

        showResearchComplete(
          task.title || 'Research Task',
          resultPreview,
          this.mainWindow || undefined
        )

        // Notify renderer to refresh tasks
        if (this.mainWindow && !this.mainWindow.isDestroyed()) {
          this.mainWindow.webContents.send('senter:research-update', {
            type: 'task_completed',
            taskId: task.id,
            title: task.title
          })
        }
      }
    } catch (error) {
      // Silent fail - daemon might be unavailable
      console.log('[TaskPoller] Check failed:', (error as Error).message)
    }
  }
}

const taskPoller = new TaskPoller()

// Register IPC handlers
export function registerSenterIPC(): void {
  console.log('[SenterIPC] Registering IPC handlers...')

  ipcMain.handle('senter:query', async (_, text: string, conversationId?: string) => {
    console.log('[SenterIPC] Received query:', text)
    const result = await senterClient.query(text, conversationId)
    console.log('[SenterIPC] Query result:', result)
    return result
  })

  ipcMain.handle('senter:research', async (_, limit?: number) => {
    console.log('[SenterIPC] Received research request, limit:', limit)
    return senterClient.getResearch(limit)
  })

  ipcMain.handle('senter:status', async () => {
    console.log('[SenterIPC] Received status request')
    return senterClient.getStatus()
  })

  ipcMain.handle('senter:health', async () => {
    console.log('[SenterIPC] Received health request')
    return senterClient.getHealth()
  })

  ipcMain.handle('senter:getConversations', async (_, conversationId?: string, includeMessages?: boolean) => {
    console.log('[SenterIPC] Received getConversations request, id:', conversationId)
    return senterClient.getConversations(conversationId, includeMessages)
  })

  ipcMain.handle('senter:saveConversation', async (_, conversation) => {
    console.log('[SenterIPC] Received saveConversation request, id:', conversation?.id)
    return senterClient.saveConversation(conversation)
  })

  ipcMain.handle('senter:getTasks', async (_, limit?: number) => {
    console.log('[SenterIPC] Received getTasks request, limit:', limit)
    return senterClient.getTasks(limit)
  })

  ipcMain.handle('senter:getJournal', async (_, date?: string, limit?: number) => {
    console.log('[SenterIPC] Received getJournal request, date:', date)
    return senterClient.getJournal(date, limit)
  })

  ipcMain.handle('senter:generateJournal', async (_, date?: string) => {
    console.log('[SenterIPC] Received generateJournal request, date:', date)
    return senterClient.generateJournal(date)
  })

  // P3-001: Context Sources
  ipcMain.handle('senter:getContextSources', async () => {
    console.log('[SenterIPC] Received getContextSources request')
    return senterClient.getContextSources()
  })

  ipcMain.handle('senter:addContextSource', async (_, source) => {
    console.log('[SenterIPC] Received addContextSource request, type:', source?.type)
    return senterClient.addContextSource(source)
  })

  ipcMain.handle('senter:removeContextSource', async (_, id: string) => {
    console.log('[SenterIPC] Received removeContextSource request, id:', id)
    return senterClient.removeContextSource(id)
  })

  // P4-001: Voice Input
  ipcMain.handle('senter:startVoiceInput', async () => {
    console.log('[SenterIPC] Received startVoiceInput request')
    return senterClient.startVoiceInput()
  })

  ipcMain.handle('senter:stopVoiceInput', async () => {
    console.log('[SenterIPC] Received stopVoiceInput request')
    return senterClient.stopVoiceInput()
  })

  ipcMain.handle('senter:getAudioStatus', async () => {
    console.log('[SenterIPC] Received getAudioStatus request')
    return senterClient.getAudioStatus()
  })

  // P4-002: Gaze Detection
  ipcMain.handle('senter:getGazeStatus', async () => {
    console.log('[SenterIPC] Received getGazeStatus request')
    return senterClient.getGazeStatus()
  })

  // V3-004: Insights (patterns, preferences, stats)
  ipcMain.handle('senter:getInsights', async (_, days?: number) => {
    console.log('[SenterIPC] Received getInsights request')
    return senterClient.getInsights(days)
  })

  // V3-005: Goals
  ipcMain.handle('senter:getGoals', async (_, status?: string, limit?: number) => {
    console.log('[SenterIPC] Received getGoals request')
    return senterClient.getGoals(status, limit)
  })

  ipcMain.handle('senter:updateGoal', async (_, goalId: string, status: string) => {
    console.log('[SenterIPC] Received updateGoal request')
    return senterClient.updateGoal(goalId, status)
  })

  console.log('[SenterIPC] IPC handlers registered')
}

export { senterClient, taskPoller }
