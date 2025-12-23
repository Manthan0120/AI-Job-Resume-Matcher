from __future__ import annotations
from typing import List, Optional, Annotated, Literal
from datetime import date
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import StrictStr
from pydantic import StringConstraints

NonEmptyStr = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]

class SectionedText(BaseModel):
    summary: str = ""
    experience: str = ""
    education: str = ""
    skills: str = ""
    other: str = ""

class ExperienceItem(BaseModel):
    title: NonEmptyStr
    company: Optional[NonEmptyStr] = None
    location: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    current: bool = False
    bullets: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_dates(self):
        if self.end_date and self.current:
            raise ValueError("Experience cannot be both current and have an end_date")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date")
        return self

class EducationItem(BaseModel):
    institution: NonEmptyStr
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    gpa: Optional[float] = Field(default=None, ge=0.0, le=4.0)

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("education end_date before start_date")
        return self

class CandidateNormalized(BaseModel):
    skills: List[NonEmptyStr] = Field(default_factory=list)
    years_experience_est: int = Field(default=0, ge=0, le=60)
    has_education: bool = False
    has_experience: bool = False

class Candidate(BaseModel):
    id: NonEmptyStr
    filename: Optional[str] = None
    content: NonEmptyStr
    sections: SectionedText
    normalized: Optional[CandidateNormalized] = None
    experiences: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    skills: List[NonEmptyStr] = Field(default_factory=list)
    type: Literal["resume"] = "resume"

    @field_validator("skills", mode="before")
    def normalize_skills(cls, v):
        # Accept list, comma/newline-delimited string, or None
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r"[,\\n]", v) if p.strip()]
            return parts
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return v

    @model_validator(mode="after")
    def _lowercase_skills(self):
        if self.skills:
            self.skills = [s.lower() for s in self.skills]
        return self

class JobFields(BaseModel):
    title: NonEmptyStr
    company: Optional[str] = "Unknown"
    location: Optional[str] = None
    description: str = ""
    requirements: str = ""
    skills: str = ""

class JobNormalized(BaseModel):
    skills: List[NonEmptyStr] = Field(default_factory=list)
    must_haves_raw: List[str] = Field(default_factory=list)

class Job(BaseModel):
    id: NonEmptyStr
    content: NonEmptyStr
    fields: JobFields
    normalized: Optional[JobNormalized] = None
    type: Literal["job"] = "job"

    @field_validator("content", mode="after")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Job content cannot be blank")
        return v