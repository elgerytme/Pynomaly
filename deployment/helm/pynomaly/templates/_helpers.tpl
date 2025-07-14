{{/*
Expand the name of the chart.
*/}}
{{- define "pynomaly.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "pynomaly.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "pynomaly.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "pynomaly.labels" -}}
helm.sh/chart: {{ include "pynomaly.chart" . }}
{{ include "pynomaly.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: pynomaly
{{- end }}

{{/*
Selector labels
*/}}
{{- define "pynomaly.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pynomaly.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "pynomaly.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "pynomaly.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URL construction
*/}}
{{- define "pynomaly.databaseURL" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.config.database.username }}:{{ .Values.secrets.database_password }}@{{ .Values.config.database.host }}:{{ .Values.config.database.port }}/{{ .Values.config.database.name }}?sslmode={{ .Values.config.database.ssl_mode }}
{{- else }}
{{- .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
Redis URL construction
*/}}
{{- define "pynomaly.redisURL" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.config.redis.ssl }}
rediss://:{{ .Values.secrets.redis_password }}@{{ .Values.config.redis.host }}:{{ .Values.config.redis.port }}/{{ .Values.config.redis.db }}
{{- else }}
redis://:{{ .Values.secrets.redis_password }}@{{ .Values.config.redis.host }}:{{ .Values.config.redis.port }}/{{ .Values.config.redis.db }}
{{- end }}
{{- else }}
{{- .Values.externalRedis.url }}
{{- end }}
{{- end }}