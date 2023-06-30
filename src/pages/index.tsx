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


      <iframe 
          src="https://ghbtns.com/github-btn.html?user=martinrgb&repo=LTC_AreaLight_R3F_Component&type=star&count=true&size=large" 
          style={{
            position: 'absolute',
            left:'16px',
            bottom: '16px',
            zIndex: 10000,
            userSelect:'none'
          }}
          width="170" height="30" title="GitHub">
          
        </iframe>
    </main>
  )
}
