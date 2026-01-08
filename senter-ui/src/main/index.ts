import { app, BrowserWindow, shell, globalShortcut } from 'electron'
import { join } from 'path'
import Store from 'electron-store'
import { registerSenterIPC, taskPoller } from './ipc/senterBridge'
import { createTray, destroyTray } from './tray'

const isDev = process.env.NODE_ENV === 'development'

// Store for persisting window position
const store = new Store<{
  windowBounds?: { x: number; y: number; width: number; height: number }
}>()

let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  // Get stored window bounds
  const bounds = store.get('windowBounds')

  mainWindow = new BrowserWindow({
    width: bounds?.width || 1200,
    height: bounds?.height || 800,
    x: bounds?.x,
    y: bounds?.y,
    minWidth: 800,
    minHeight: 600,
    show: false,
    autoHideMenuBar: true,
    // macOS specific - frameless with vibrancy
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 20, y: 20 },
    transparent: true,
    vibrancy: 'under-window',
    visualEffectState: 'active',
    backgroundColor: '#00000000',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  // Save window position on close
  mainWindow.on('close', () => {
    if (mainWindow) {
      store.set('windowBounds', mainWindow.getBounds())
    }
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // Load the renderer
  if (isDev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  // Create tray icon
  createTray(mainWindow)

  // P2-005: Start task completion polling for notifications
  taskPoller.setMainWindow(mainWindow)
  taskPoller.start()
}

// Register global keyboard shortcut
function registerGlobalShortcut(): void {
  // Cmd+Shift+S to show/hide window
  globalShortcut.register('CommandOrControl+Shift+S', () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide()
      } else {
        mainWindow.show()
        mainWindow.focus()
      }
    }
  })
}

app.whenReady().then(() => {
  // Set app user model id for windows
  app.setAppUserModelId('com.senter.ai')

  // Register IPC handlers for Senter backend
  registerSenterIPC()

  // Register global shortcut
  registerGlobalShortcut()

  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('will-quit', () => {
  // Unregister all shortcuts
  globalShortcut.unregisterAll()
  // Stop task poller
  taskPoller.stop()
  // Destroy tray
  destroyTray()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
