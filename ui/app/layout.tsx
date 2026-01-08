import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Senter',
  description: 'Your local AI assistant powered by Ollama',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark" style={{ background: 'transparent' }}>
      <body className="antialiased" style={{ background: 'transparent' }}>{children}</body>
    </html>
  )
}
