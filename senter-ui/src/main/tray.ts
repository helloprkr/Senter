import { Tray, Menu, nativeImage, BrowserWindow, app } from 'electron'

let tray: Tray | null = null

export function createTray(mainWindow: BrowserWindow): Tray {
  // Create tray icon (diamond shape using text for now)
  const icon = nativeImage.createEmpty()

  tray = new Tray(icon)
  tray.setTitle('◇') // Unicode diamond

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show Senter',
      click: () => {
        if (mainWindow.isVisible()) {
          mainWindow.hide()
        } else {
          mainWindow.show()
          mainWindow.focus()
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Status: Healthy',
      enabled: false,
    },
    { type: 'separator' },
    {
      label: 'Settings',
      click: () => {
        mainWindow.show()
        mainWindow.webContents.send('open-settings')
      },
    },
    { type: 'separator' },
    {
      label: 'Quit Senter',
      click: () => {
        app.quit()
      },
    },
  ])

  tray.setContextMenu(contextMenu)

  // Click to show/hide
  tray.on('click', () => {
    if (mainWindow.isVisible()) {
      mainWindow.hide()
    } else {
      mainWindow.show()
      mainWindow.focus()
    }
  })

  return tray
}

export function updateTrayTitle(unreadCount: number, isResearching: boolean): void {
  if (!tray) return

  let title = '◇' // Default idle state
  if (isResearching) {
    title = '◆' // Filled diamond when researching
  } else if (unreadCount > 0) {
    title = `● ${unreadCount}` // Circle with count for new findings
  }

  tray.setTitle(title)
}

export function destroyTray(): void {
  if (tray) {
    tray.destroy()
    tray = null
  }
}
