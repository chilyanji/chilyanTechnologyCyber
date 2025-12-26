"use client"
import Link from "next/link"

interface NavigationProps {
  onTabChange?: (tab: string) => void
}

export default function Navigation({ onTabChange }: NavigationProps) {
  return (
    <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary-dark rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">PD</span>
            </div>
            <span className="font-bold text-lg text-foreground hidden sm:inline">Phishing Detection</span>
          </Link>

          <div className="flex items-center gap-1">
            <button className="px-4 py-2 rounded-lg hover:bg-border transition text-foreground text-sm font-medium">
              Dashboard
            </button>
            <button className="px-4 py-2 rounded-lg hover:bg-border transition text-foreground text-sm font-medium">
              Analyze
            </button>
            <button className="px-4 py-2 rounded-lg hover:bg-border transition text-foreground text-sm font-medium">
              History
            </button>
            <button className="px-4 py-2 rounded-lg hover:bg-border transition text-foreground text-sm font-medium">
              Settings
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}
