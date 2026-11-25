import { useLocation } from 'react-router-dom';
import { useAppStore } from '../store/appStore';

const TITLES = {
  '/': 'Dashboard',
  '/index-repo': 'Index a Repo',
  '/code-memory': 'Code Memory',
  '/repositories': 'Index a Repo',
  '/ask': 'Ask Codebase',
  '/plan': 'Plan Task',
  '/fix': 'Fix a Bug',
  '/execute': 'Execute Workflow',
  '/activity': 'Activity Log',
};

export default function Topbar() {
  const { pathname } = useLocation();
  const { currentRepo } = useAppStore();
  const title = TITLES[pathname] || 'Dashboard';
  const version = currentRepo ? (currentRepo.commit ? currentRepo.commit.slice(0, 7) : currentRepo.ref) : '';

  return (
    <div className="topbar">
      <div className="topbar-left">
        <span className="topbar-title">{title}</span>
        {currentRepo && (
          <>
            <span className="topbar-sep">/</span>
            <span className="topbar-repo">
              {currentRepo.repo}
              {version ? ` @ ${version}` : ''}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
