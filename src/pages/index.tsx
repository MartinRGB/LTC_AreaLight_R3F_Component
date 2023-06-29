import Image from 'next/image'
import { Inter } from 'next/font/google'
import { Effect } from '../effects/Effect'
const inter = Inter({ subsets: ['latin'] })

export default function Home() {
  return (
    <main
      className={`w-full h-full ${inter.className}`}
    >
      <Effect className="w-full h-full">

      </Effect>
    </main>
  )
}
