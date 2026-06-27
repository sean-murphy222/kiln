/**
 * Mappers from Forge API client types to Forge store types.
 *
 * The store types carry a few display-only fields (e.g. `competency_name`,
 * `contributor_name`, coverage counts) that the raw API records do not
 * include. These helpers translate API records into store records, accepting
 * optional enrichment values resolved by the caller.
 */

import type {
  Discipline as ApiDiscipline,
  Competency as ApiCompetency,
  Example as ApiExample,
} from "@/api/forge";
import type {
  ForgeDiscipline,
  ForgeCompetency,
  ForgeExample,
} from "@/store/useForgeStore";

export function mapDiscipline(discipline: ApiDiscipline): ForgeDiscipline {
  return {
    id: discipline.id,
    name: discipline.name,
    description: discipline.description,
    status: discipline.status,
    competency_count: 0,
    example_count: 0,
    coverage_pct: 0,
  };
}

export function mapCompetency(competency: ApiCompetency): ForgeCompetency {
  return {
    id: competency.id,
    discipline_id: competency.discipline_id,
    name: competency.name,
    description: competency.description,
    level: competency.level,
    parent_id: competency.parent_id,
    example_count: competency.example_count,
    target_count: competency.target_count,
  };
}

export function mapExample(
  example: ApiExample,
  competencyName: string,
): ForgeExample {
  return {
    id: example.id,
    competency_id: example.competency_id,
    competency_name: competencyName,
    question: example.question,
    answer: example.answer,
    context: example.context,
    status: example.status,
    contributor_id: example.contributor_id,
    contributor_name: example.contributor_id,
    created_at: example.created_at,
  };
}
