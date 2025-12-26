"use client"

import { useState } from "react"
import Dashboard from "@/components/dashboard"
import DetectionForm from "@/components/detection-form"
import Navigation from "@/components/navigation"

export default function Home() {
  const [activeTab, setActiveTab] = useState("dashboard")

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main className="container mx-auto px-4 py-8">
        {activeTab === "dashboard" ? <Dashboard /> : <DetectionForm />}
      </main>
    </div>
  )
}
