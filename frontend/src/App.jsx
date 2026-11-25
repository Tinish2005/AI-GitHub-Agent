import React from 'react';
import { Routes, Route } from 'react-router-dom';
import './styles/agent.css';
import { AppStoreProvider } from './store/appStore';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import ServerGate from './components/ServerGate';
import Home from './pages/Home';
import Ask from './pages/Ask';
import Plan from './pages/Plan';
import Fix from './pages/Fix';
import Execute from './pages/Execute';
import IndexRepo from './pages/IndexRepo';
import CodeMemory from './pages/CodeMemory';
import Activity from './pages/Activity';

export default function App() {
  return (
    <AppStoreProvider>
      <div className="app">
        <Sidebar />
        <div className="main">
          <Topbar />
          <div className="content">
            <ServerGate>
            <Routes>
              {/* Dashboard */}
              <Route path="/" element={<Home />} />

              {/* Workspace Dedicated Pages */}
              <Route path="/index-repo" element={<IndexRepo />} />
              <Route path="/repositories" element={<IndexRepo />} />
              <Route path="/ask" element={<Ask />} />

              {/* Workflow Dedicated Pages */}
              <Route path="/plan" element={<Plan />} />
              <Route path="/fix" element={<Fix />} />
              <Route path="/execute" element={<Execute />} />

              {/* Insights */}
              <Route path="/code-memory" element={<CodeMemory />} />
              <Route path="/activity" element={<Activity />} />
            </Routes>
            </ServerGate>
          </div>
        </div>
      </div>
    </AppStoreProvider>
  );
}
