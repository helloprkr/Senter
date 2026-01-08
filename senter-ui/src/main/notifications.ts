import { Notification, BrowserWindow } from 'electron'

interface NotificationOptions {
  title: string
  body: string
  mainWindow?: BrowserWindow
}

export function showNotification({ title, body, mainWindow }: NotificationOptions): void {
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
    }
  })

  notification.show()
}

export function showResearchComplete(topic: string, summary: string, mainWindow?: BrowserWindow): void {
  showNotification({
    title: `Research Complete: ${topic}`,
    body: summary.length > 100 ? summary.substring(0, 100) + '...' : summary,
    mainWindow,
  })
}

export function showNewFindings(count: number, mainWindow?: BrowserWindow): void {
  showNotification({
    title: 'New Research Findings',
    body: `Senter has ${count} new research result${count > 1 ? 's' : ''} for you`,
    mainWindow,
  })
}
