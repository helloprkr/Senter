import { motion } from 'framer-motion'

export function RainbowBorder() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute inset-0 rounded-3xl p-[2px] overflow-hidden"
    >
      <div
        className="absolute inset-0 rounded-3xl"
        style={{
          background: `linear-gradient(
            45deg,
            #ff0000,
            #ff7f00,
            #ffff00,
            #00ff00,
            #0000ff,
            #4b0082,
            #9400d3,
            #ff0000
          )`,
          backgroundSize: '400% 400%',
          animation: 'rainbow 2s linear infinite',
        }}
      />
      <div className="absolute inset-[2px] rounded-3xl bg-brand-dark" />
    </motion.div>
  )
}
