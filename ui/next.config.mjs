/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export for Electron standalone app
  output: 'export',

  // Disable trailing slashes for cleaner file paths
  trailingSlash: false,

  // Disable Next.js dev indicator (the floating "N" bubble)
  devIndicators: false,

  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
