import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'YummyMusic Video Generator',
  description: 'Generate amazing videos with AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
