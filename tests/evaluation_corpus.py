"""
An uncurated-ish resume/job corpus for tests/run_evaluation.py.

Unlike tests/compare_agentic_vs_pipeline.py's 3 pairs (each deliberately
engineered to elicit one specific mechanism -- a demonstration, not evidence),
these 16 resumes and 16 jobs were written independently, for plausible
variety across roles, without engineering phrasing gaps, false negatives, or
any other property meant to trigger a specific escape hatch. Some resumes
happen to be terse, some verbose, some list skills as a flat line, some
describe them only in prose -- that variation is incidental (how real
resumes vary), not constructed to prove a point.

This doesn't eliminate authorship bias (I wrote all of it, so I'm not a
neutral third party), but it's a meaningfully different exercise from writing
"a resume that hides Kubernetes behind vague prose" on purpose.
"""

JOBS = [
    {"id": "job_data_eng", "title": "Data Engineer", "company": "Northline Analytics", "type": "job", "content": (
        "Title: Data Engineer\n\nDescription: Build and maintain ETL pipelines feeding our analytics "
        "warehouse. Own data quality and pipeline reliability.\n\n"
        "Requirements: Python, SQL, Airflow, dbt, AWS (S3, Redshift), 3+ years experience with "
        "production data pipelines."
    )},
    {"id": "job_frontend", "title": "Frontend Engineer", "company": "Lucent UI Co.", "type": "job", "content": (
        "Title: Frontend Engineer\n\nDescription: Build our customer-facing dashboard.\n\n"
        "Requirements: React, TypeScript, CSS, REST API integration, component testing, 2+ years."
    )},
    {"id": "job_backend_node", "title": "Backend Engineer (Node.js)", "company": "Fetchwell", "type": "job", "content": (
        "Title: Backend Engineer\n\nDescription: Own our order-processing API and its uptime.\n\n"
        "Requirements: Node.js, Express, PostgreSQL, REST APIs, Docker, CI/CD."
    )},
    {"id": "job_ml_eng", "title": "Machine Learning Engineer", "company": "Verdant AI", "type": "job", "content": (
        "Title: Machine Learning Engineer\n\nDescription: Take models from notebook to production serving "
        "real-time predictions.\n\nRequirements: Python, PyTorch or TensorFlow, model deployment, Docker, "
        "experience with feature pipelines."
    )},
    {"id": "job_devops", "title": "DevOps/SRE", "company": "Ridgeline Systems", "type": "job", "content": (
        "Title: DevOps/SRE\n\nDescription: Keep our infrastructure reliable and our deploys boring.\n\n"
        "Requirements: Kubernetes, Terraform, AWS, on-call experience, CI/CD pipelines, monitoring/alerting."
    )},
    {"id": "job_qa", "title": "QA Engineer", "company": "Bracket Software", "type": "job", "content": (
        "Title: QA Engineer\n\nDescription: Design and run our automated test suite across web and API "
        "surfaces.\n\nRequirements: Selenium or Playwright, Python or JavaScript, CI integration, "
        "test case design, bug triage experience."
    )},
    {"id": "job_ios", "title": "iOS Developer", "company": "Palette Mobile", "type": "job", "content": (
        "Title: iOS Developer\n\nDescription: Ship features in our consumer iOS app used by millions.\n\n"
        "Requirements: Swift, UIKit or SwiftUI, Xcode, App Store release process, 2+ years mobile experience."
    )},
    {"id": "job_pm", "title": "Product Manager", "company": "Northline Analytics", "type": "job", "content": (
        "Title: Product Manager\n\nDescription: Own the roadmap for our analytics product line, working "
        "closely with engineering and design.\n\nRequirements: 3+ years PM experience, experience shipping "
        "B2B SaaS features, comfortable reading basic SQL, stakeholder management."
    )},
    {"id": "job_ux", "title": "UX Designer", "company": "Lucent UI Co.", "type": "job", "content": (
        "Title: UX Designer\n\nDescription: Design end-to-end flows for our dashboard product.\n\n"
        "Requirements: Figma, user research experience, design systems, prototyping, portfolio required."
    )},
    {"id": "job_data_sci", "title": "Data Scientist", "company": "Verdant AI", "type": "job", "content": (
        "Title: Data Scientist\n\nDescription: Analyze user behavior data and build predictive models "
        "to guide product decisions.\n\nRequirements: Python, pandas, statistics, A/B testing experience, "
        "SQL, communicating findings to non-technical stakeholders."
    )},
    {"id": "job_security", "title": "Security Engineer", "company": "Ridgeline Systems", "type": "job", "content": (
        "Title: Security Engineer\n\nDescription: Harden our infrastructure and lead incident response.\n\n"
        "Requirements: Threat modeling, penetration testing exposure, AWS security (IAM, VPC), "
        "vulnerability management, scripting (Python or Bash)."
    )},
    {"id": "job_tech_writer", "title": "Technical Writer", "company": "Bracket Software", "type": "job", "content": (
        "Title: Technical Writer\n\nDescription: Write and maintain developer-facing API documentation.\n\n"
        "Requirements: Experience documenting REST APIs, Markdown, working with engineers to keep docs "
        "current, plain-English writing skill."
    )},
    {"id": "job_sales_eng", "title": "Sales Engineer", "company": "Fetchwell", "type": "job", "content": (
        "Title: Sales Engineer\n\nDescription: Run technical demos and proofs-of-concept for enterprise "
        "prospects.\n\nRequirements: Comfortable presenting to technical and non-technical audiences, "
        "API integration experience, some scripting ability, customer-facing experience."
    )},
    {"id": "job_dba", "title": "Database Administrator", "company": "Palette Mobile", "type": "job", "content": (
        "Title: Database Administrator\n\nDescription: Own performance, backups, and reliability for our "
        "production PostgreSQL fleet.\n\nRequirements: PostgreSQL internals, query optimization, backup/"
        "recovery procedures, on-call rotation experience."
    )},
    {"id": "job_cloud_arch", "title": "Cloud Architect", "company": "Verdant AI", "type": "job", "content": (
        "Title: Cloud Architect\n\nDescription: Design our multi-region AWS architecture as we scale.\n\n"
        "Requirements: Deep AWS experience (multi-account, networking), Terraform, cost optimization, "
        "architecture review experience, 5+ years."
    )},
    {"id": "job_fullstack", "title": "Full-stack Developer", "company": "Bracket Software", "type": "job", "content": (
        "Title: Full-stack Developer\n\nDescription: Work across our React frontend and Python backend "
        "roughly evenly.\n\nRequirements: React, Python, REST APIs, SQL, comfortable owning a feature "
        "end to end."
    )},
]

# One resume per job above, same order, written independently -- not
# engineered to match or mismatch any particular job's phrasing.
RESUMES = [
    {"label": "data_eng_resume", "paired_job_id": "job_data_eng", "content": (
        "Maria Chen\nData engineer with 4 years building pipelines for analytics teams. Comfortable "
        "owning a pipeline from ingestion to warehouse. Recent work: rebuilt a nightly ETL job in "
        "Airflow that was failing intermittently, cutting failures to near zero. Wrote most of our "
        "dbt models for the finance reporting layer. Day to day: Python, SQL, some AWS (mostly S3 and "
        "Redshift access, not deep infra work). Comfortable with Git and code review."
    )},
    {"label": "frontend_resume", "paired_job_id": "job_frontend", "content": (
        "Devon Marsh\nFrontend developer, 3 years. Built and maintained a customer dashboard used by "
        "several thousand paying accounts. Worked mostly in React with TypeScript; wrote component "
        "tests with Jest and Testing Library. Integrated with a REST backend for most features. "
        "Comfortable with CSS and responsive layout; less experience with design systems specifically."
    )},
    {"label": "backend_node_resume", "paired_job_id": "job_backend_node", "content": (
        "Priya Nair\nBackend engineer. Built an order-processing service in Node.js and Express backed "
        "by Postgres, handling several thousand orders a day. Wrote the CI pipeline that runs tests and "
        "deploys via Docker on merge to main. On-call for the service for the last year. Comfortable "
        "reading and writing SQL migrations."
    )},
    {"label": "ml_eng_resume", "paired_job_id": "job_ml_eng", "content": (
        "Sam Okafor\nML engineer, came from a research background, 2 years in industry. Took a "
        "recommendation model from a research notebook to a served endpoint handling production "
        "traffic, using PyTorch and a simple Flask wrapper, later moved to a proper serving setup. "
        "Built the feature pipeline that feeds the model. Comfortable with Docker for packaging."
    )},
    {"label": "devops_resume", "paired_job_id": "job_devops", "content": (
        "Jordan Reyes\nInfrastructure engineer. Migrated a fleet of services from manually-managed VMs "
        "to Kubernetes, writing the Terraform to provision the underlying AWS resources. On-call "
        "rotation owner for 18 months. Built most of our alerting rules from scratch after a bad outage "
        "taught us we had none. CI/CD pipelines are mine end to end."
    )},
    {"label": "qa_resume", "paired_job_id": "job_qa", "content": (
        "Ava Whitman\nQA engineer, 3 years. Built an end-to-end test suite for a web app using "
        "Playwright, wired into CI so a failing test blocks merge. Also own manual test case design for "
        "new features and triage incoming bug reports before they reach engineering. Some Python "
        "scripting for test utilities."
    )},
    {"label": "ios_resume", "paired_job_id": "job_ios", "content": (
        "Miguel Alvarez\niOS developer, 4 years, shipped features in a consumer app with a large "
        "install base. Comfortable in Swift and SwiftUI, some legacy UIKit code still in the app. Has "
        "gone through the App Store review process many times, including a couple of rejections that "
        "needed real fixes, not just resubmission."
    )},
    {"label": "pm_resume", "paired_job_id": "job_pm", "content": (
        "Lena Fischer\nProduct manager, 5 years, most recently on a B2B analytics product. Owned the "
        "roadmap for a reporting feature area, working directly with engineering and design leads. "
        "Comfortable writing basic SQL to answer my own questions rather than waiting on an analyst. "
        "Regularly present to executive stakeholders."
    )},
    {"label": "ux_resume", "paired_job_id": "job_ux", "content": (
        "Noah Kim\nUX designer, 3 years. Designed the end-to-end flow for a dashboard redesign, from "
        "research through high-fidelity prototypes in Figma. Maintains our component design system. "
        "Has run and synthesized user interviews for two major feature launches."
    )},
    {"label": "data_sci_resume", "paired_job_id": "job_data_sci", "content": (
        "Grace Liu\nData scientist, 3 years. Ran A/B tests for a subscription product and built the "
        "dashboards the product team used to make ship/no-ship calls. Comfortable in Python and pandas "
        "for analysis, writes most of my own SQL against the warehouse. Regularly presents findings to "
        "non-technical product and design partners."
    )},
    {"label": "security_resume", "paired_job_id": "job_security", "content": (
        "Tariq Hassan\nSecurity engineer, 4 years. Led incident response for two significant security "
        "events, including post-incident writeups. Comfortable with AWS IAM and VPC hardening. Has done "
        "internal penetration testing exercises but not formally certified. Writes Python and Bash "
        "tooling for vulnerability scanning."
    )},
    {"label": "tech_writer_resume", "paired_job_id": "job_tech_writer", "content": (
        "Ellie Sandoval\nTechnical writer, 3 years, focused entirely on developer-facing documentation "
        "for a REST API product. Works directly with engineers to keep docs in sync with releases. "
        "Writes in Markdown, publishes through a static site generator. Prior background as a software "
        "engineer before moving into writing full time."
    )},
    {"label": "sales_eng_resume", "paired_job_id": "job_sales_eng", "content": (
        "Marcus Webb\nSales engineer, 2 years. Runs technical demos and builds proof-of-concept "
        "integrations for enterprise prospects during the sales cycle. Comfortable presenting to both "
        "engineering leads and executives in the same meeting. Writes basic integration scripts against "
        "customer APIs when a demo needs it."
    )},
    {"label": "dba_resume", "paired_job_id": "job_dba", "content": (
        "Renata Silva\nDatabase administrator, 6 years, owns a production PostgreSQL fleet supporting a "
        "high-traffic app. Handles query optimization for slow endpoints flagged by the app team, and "
        "owns backup and recovery procedures, tested via regular restore drills. On-call for database "
        "incidents."
    )},
    {"label": "cloud_arch_resume", "paired_job_id": "job_cloud_arch", "content": (
        "David Okonkwo\nCloud architect, 7 years. Designed a multi-account AWS architecture during a "
        "company-wide infrastructure overhaul, with a strong focus on network segmentation and cost "
        "control. Leads architecture review for new infrastructure proposals. Terraform for all "
        "provisioning."
    )},
    {"label": "fullstack_resume", "paired_job_id": "job_fullstack", "content": (
        "Sofia Petrov\nFull-stack developer, 3 years, splits time roughly evenly between a React "
        "frontend and a Python backend for an internal tools product. Comfortable owning a feature from "
        "database schema through the UI. Writes SQL migrations and REST endpoints, then builds the React "
        "views that consume them."
    )},
]
