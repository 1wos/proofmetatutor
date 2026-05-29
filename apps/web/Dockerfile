# Next.js (standalone) image for Cloud Run.
# NEXT_PUBLIC_* must be present at build time, so the API URL is a build arg.
FROM node:20-slim AS builder
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm install --no-audit --no-fund --legacy-peer-deps
COPY apps/web/ ./
ARG NEXT_PUBLIC_VERIFIER_API=""
ENV NEXT_PUBLIC_VERIFIER_API=${NEXT_PUBLIC_VERIFIER_API}
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=8080
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 8080
CMD ["node", "server.js"]
