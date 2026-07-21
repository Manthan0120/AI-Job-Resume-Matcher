import { BrowserRouter, Routes, Route } from "react-router-dom";
import Shell from "./components/Shell";
import DataManagement from "./pages/DataManagement";
import ResumeToJobs from "./pages/ResumeToJobs";
import JobToResumes from "./pages/JobToResumes";
import AgenticMatch from "./pages/AgenticMatch";
import Chat from "./pages/Chat";
import Analytics from "./pages/Analytics";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<DataManagement />} />
          <Route path="resume-to-jobs" element={<ResumeToJobs />} />
          <Route path="job-to-resumes" element={<JobToResumes />} />
          <Route path="agent" element={<AgenticMatch />} />
          <Route path="chat" element={<Chat />} />
          <Route path="analytics" element={<Analytics />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
