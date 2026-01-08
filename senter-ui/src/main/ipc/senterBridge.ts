import { ipcMain } from 'electron'
import * as net from 'net'

const SOCKET_PATH = '/tmp/senter.sock'
const CONNECTION_TIMEOUT = 10000

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
}

const senterClient = new SenterClient()

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

  console.log('[SenterIPC] IPC handlers registered')
}

export { senterClient }
