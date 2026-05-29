import "./globals.css";

export const metadata = {
  title: "ProofMetaTutor",
  description: "Evidence-traced Korean math tutoring prototype"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

