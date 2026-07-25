import { NavLink, Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Ask from "./pages/Ask.jsx";
import Plan from "./pages/Plan.jsx";
import Fix from "./pages/Fix.jsx";
import Execute from "./pages/Execute.jsx";
import IndexRepo from "./pages/IndexRepo.jsx";

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">AI <span>GitHub Agent</span></div>
        <nav className="nav">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/index-repo">Index a Repo</NavLink>
          <NavLink to="/ask">Ask the Codebase</NavLink>
          <NavLink to="/plan">Plan</NavLink>
          <NavLink to="/fix">Propose Fix</NavLink>
          <NavLink to="/execute">Execute</NavLink>
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/index-repo" element={<IndexRepo />} />
          <Route path="/ask" element={<Ask />} />
          <Route path="/plan" element={<Plan />} />
          <Route path="/fix" element={<Fix />} />
          <Route path="/execute" element={<Execute />} />
        </Routes>
      </main>
    </div>
  );
}