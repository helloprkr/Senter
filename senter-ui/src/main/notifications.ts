import { Notification, BrowserWindow } from 'electron'

interface NotificationOptions {
  title: string
  body: string
  mainWindow?: BrowserWindow
  taskId?: string  // V3-003: Deep link to task
}

export function showNotification({ title, body, mainWindow, taskId }: NotificationOptions): void {
  // Check if notifications are supported
  if (!Notification.isSupported()) {
    console.warn('Notifications are not supported on this system')
    return
  }

  const notification = new Notification({
    title,
    body,
    silent: false,
  })

  notification.on('click', () => {
    if (mainWindow) {
      mainWindow.show()
      mainWindow.focus()
      // V3-003: Navigate to specific task when notification clicked
      if (taskId) {
        mainWindow.webContents.send('senter:navigate-to-task', { taskId })
      }
    }
  })

  notification.show()
}

// V3-003: Enhanced research complete notification with task ID for deep linking
export function showResearchComplete(
  topic: string,
  summary: string,
  mainWindow?: BrowserWindow,
  taskId?: string,
  sourceCount?: number
): void {
  // V3-003: Include source count in notification if available
  let body = summary.length > 100 ? summary.substring(0, 100) + '...' : summary
  if (sourceCount !== undefined && sourceCount > 0) {
    body = `Found ${sourceCount} source${sourceCount > 1 ? 's' : ''}. ${body}`
  }

  showNotification({
    title: `Research Complete: ${topic}`,
    body,
    mainWindow,
    taskId,
  })
}

export function showNewFindings(count: number, mainWindow?: BrowserWindow): void {
  showNotification({
    title: 'New Research Findings',
    body: `Senter has ${count} new research result${count > 1 ? 's' : ''} for you`,
    mainWindow,
  })
}
